import os
import ollama
from typing import List
from pdf2image import convert_from_path

# --- CONFIGURATION ---

# CRITICAL: The Instructor MUST use the Vision-Language (VL) version to see the PDF.
# 'qwen3:4b' is text-only and will fail to read the manual images.
VISION_MODEL = "llava:13b" 

# The User Agent is text-only, so we use the standard model to be efficient.
TEXT_MODEL = "gpt-oss:20b"

# --- HELPER FUNCTIONS ---

def save_temp_images(pdf_path: str) -> List[str]:
    """
    Converts a PDF file into a list of temporary JPEG images.
    """
    # Robust check: Add .pdf extension if user forgot it
    if not os.path.exists(pdf_path) and os.path.exists(pdf_path + ".pdf"):
        pdf_path += ".pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Cannot find file at: {pdf_path}")

    print(f"Converting '{pdf_path}' to images...")
    
    # pdf2image requires poppler installed (brew install poppler)
    images = convert_from_path(pdf_path)
    image_paths = []
    
    os.makedirs("temp_pages", exist_ok=True)
    
    for i, image in enumerate(images):
        path = f"temp_pages/page_{i+1}.jpg"
        image.save(path, "JPEG")
        image_paths.append(path)
    
    return image_paths

# --- AGENT CLASSES ---

class InstructorAgent:
    """
    The 'Expert' Agent (Vision-Enabled).
    Uses Qwen 3 VL to see the manual.
    """
    def __init__(self):
        # Qwen 3 (2025) prompt engineering
        self.cot_system_prompt = (
            "You are an expert furniture assembly instructor. "
            "Your goal is to describe assembly steps clearly based on the provided diagram. "
            "Before generating the final instruction, follow this process:\n"
            "1. List all objects visible in the diagram.\n"
            "2. Compare their sizes/shapes to identify specific part numbers.\n"
            "3. Identify potential ambiguities (e.g., 'Do not confuse Bolt A with Bolt B').\n"
            "4. Only THEN, write the Final Instruction.\n\n"
            "Output format: 'THOUGHTS: [Your reasoning] \n FINAL INSTRUCTION: [The clear text]'"
        )

        # Self-Correction Prompt
        self.reflection_prompt = (
            "You are a strict quality control critic. "
            "Look at this draft instruction. Is there ANY part that a beginner might misunderstand? "
            "Is the part number missing? Is the orientation clear? "
            "If it is vague, rewrite it to be bulletproof. "
            "If it is perfect, output it exactly as is."
        )

    def generate_instruction(self, image_path: str, context: str = "") -> str:
        # Step 1: Vision analysis using qwen3-vl:4b
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {'role': 'system', 'content': self.cot_system_prompt},
                {'role': 'user', 'content': f"Analyze this manual page. {context}", 'images': [image_path]}
            ],
            options={'temperature': 0.2} 
        )
        
        raw_output = response['message']['content']
        
        # Parse output
        if "FINAL INSTRUCTION:" in raw_output:
            draft_instruction = raw_output.split("FINAL INSTRUCTION:")[1].strip()
        else:
            draft_instruction = raw_output 

        # Step 2: Self-Reflection (Can use standard text model here)
        return self._reflect_and_refine(draft_instruction)

    def _reflect_and_refine(self, draft: str) -> str:
        # We use the text-only model for this logical check to save VRAM
        response = ollama.chat(
            model=TEXT_MODEL,
            messages=[
                {'role': 'system', 'content': self.reflection_prompt},
                {'role': 'user', 'content': f"Draft Instruction: \"{draft}\"\n\nImprove this if necessary."}
            ],
            options={'temperature': 0.1}
        )
        return response['message']['content']

    def answer_clarification(self, image_path: str, current_instruction: str, user_question: str) -> str:
        # Visual re-check needs the VL model
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {'role': 'system', 'content': "You are a helpful expert. The user is confused. Provide specific details to clarify."},
                {'role': 'user', 'content': f"Previous Instruction: '{current_instruction}'. User Question: '{user_question}'. Provide a detailed answer.", 'images': [image_path]}
            ]
        )
        return response['message']['content']


class UserAgent:
    """
    The 'Novice' Agent (Text-Only).
    Uses standard Qwen 3 (text) to critique instructions.
    """
    def __init__(self):
        self.system_prompt = (
            "You are a cautious novice builder. "
            "You cannot see diagrams, only text. "
            "Critique instructions aggressively for clarity."
        )

    def evaluate(self, instruction_text: str) -> str:
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': "Instruction: 'Attach the legs to the frame.'"},
            {'role': 'assistant', 'content': "CLARIFICATION_NEEDED: Which legs? Where on the frame do they go?"},
            {'role': 'user', 'content': "Instruction: 'Take the two front legs (Part A) and screw them into the front holes using Screw 102.'"},
            {'role': 'assistant', 'content': "SATISFIED"},
            {'role': 'user', 'content': f"Instruction: \"{instruction_text}\""}
        ]

        response = ollama.chat(
            model=TEXT_MODEL,
            messages=messages,
            options={'temperature': 0.5}
        )
        return response['message']['content']

# --- MAIN ORCHESTRATOR ---

def process_manual(pdf_path: str):
    instructor = InstructorAgent()
    user = UserAgent()
    
    try:
        page_images = save_temp_images(pdf_path)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        print("Tip: Make sure you installed poppler: 'brew install poppler'")
        return

    final_manual_text = []

    print(f"\n--- STARTING QWEN 3 TRANSLATION ({VISION_MODEL}) ---\n")

    for i, img_path in enumerate(page_images):
        print(f"\nProcessing Page {i+1}...")
        
        current_instruction = instructor.generate_instruction(img_path, context="Identify assembly steps.")
        
        if "no assembly steps" in current_instruction.lower():
            print("  (Skipping page: No steps detected)")
            continue

        print(f"  [Instructor Proposed]: {current_instruction[:100]}...")

        negotiation_active = True
        loop_count = 0
        
        while negotiation_active and loop_count < 3:
            feedback = user.evaluate(current_instruction)
            
            if "SATISFIED" in feedback:
                print(f"  [User Agent]: {feedback}")
                final_manual_text.append(f"--- Page {i+1} ---\n{current_instruction}\n")
                negotiation_active = False
            
            elif "CLARIFICATION_NEEDED" in feedback:
                question = feedback.replace("CLARIFICATION_NEEDED:", "").strip()
                print(f"  [User Agent Asked]: {question}")
                current_instruction = instructor.answer_clarification(img_path, current_instruction, question)
                print(f"  [Instructor Clarified]: {current_instruction[:100]}...")
                loop_count += 1
            else:
                print(f"  [User Agent Ambiguous]: {feedback}")
                negotiation_active = False

    print("\n\n=== FINAL MANUAL ===\n")
    print("\n".join(final_manual_text))
    
    # Cleanup
    for img in page_images:
        os.remove(img)
    os.rmdir("temp_pages")

if __name__ == "__main__":
    # Your file path
    target_file = r".\pdfs\ikea2.pdf"
    
    process_manual(target_file)