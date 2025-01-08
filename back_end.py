from flask import Flask, render_template, request
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader  # Update this to langchain_community.document_loaders in production
import tempfile
import os

app = Flask(__name__)

# Load models
qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ur")
question_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

# Global variables
chunk_size = 1024  # Global chunk size for processing
max_length = 100    # Global max length for question generation
max_new_tokens = 50 # Number of new tokens for question generation

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global chunk_size  # Declare chunk_size as global
    option = request.form['option']
    pdf_file = request.files['pdf_file']
    
    if pdf_file:
        # Use NamedTemporaryFile to handle the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        # Load the PDF content
        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()
        full_text = " ".join([page.page_content for page in data])

        if option == "Question and Answer":
            user_query = request.form['user_query']
            answer = qa_model(question=user_query, context=full_text)['answer']
            return render_template('index.html', answer=answer)

        elif option == "Summarize PDF":
            summary = summarization_model(full_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            return render_template('index.html', summary=summary)

        elif option == "Translate PDF":
            # Adjusted to a smaller size to fit model constraints
            chunk_size = 300  # If you want to adjust for translation
            chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
            translations = []

            try:
                # Process each chunk
                for idx, chunk in enumerate(chunks):
                    print(f"Translating chunk {idx + 1}/{len(chunks)}...")
                    translation = translation_model(chunk, truncation=True, max_length=512)[0]['translation_text']
                    translations.append(translation)
                
                translated_text = " ".join(translations)
                return render_template('index.html', translation=translated_text)

            except Exception as e:
                error_message = f"An error occurred during translation: {str(e)}"
                print(error_message)  # Log the error for debugging
                return render_template('index.html', error=error_message)

        elif option == "Generate Questions":
            chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
            questions = []

            for idx, chunk in enumerate(chunks):
                print(f"Processing chunk {idx + 1}/{len(chunks)}...")
                prompt = f"Based on the following text, generate meaningful questions:\n\n{chunk}\n\nQuestions:"
                output = question_generator(prompt, max_length=max_length + len(prompt.split()), 
                                            max_new_tokens=max_new_tokens, num_return_sequences=1)

                generated_text = output[0]['generated_text'].strip()
                generated_questions = generated_text.split('\n')
                
                for question in generated_questions:
                    question = question.strip()
                    if question and question not in questions:
                        questions.append(question)

            questions_output = "\n".join(questions)
            return render_template('index.html', questions=questions_output)

        os.remove(tmp_file_path)  # Clean Temp File

    return render_template('index.html', error="No file uploaded.")

if __name__ == '__main__':
    app.run(debug=True)