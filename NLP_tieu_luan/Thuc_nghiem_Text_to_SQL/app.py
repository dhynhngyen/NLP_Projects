import torch 
import gradio as gr 
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


#1. Load model va tokenizer 

MODEL_DIR = Path(__file__).parent / 'outputs'/'t5_spider_text2sql'/'checkpoint-2190'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()

#2. Chuẩn bị Input cho Model

def build_input(question, schema):
    #format bên dưới là format dùng khi tạo thành dataset
    return f"Question: {question}\nSchema: {schema}"


#3. Hàm dự đoán SQL

def predict_sql(question: str, schema):
    if not question.strip():
        return "Vui lòng nhập câu hỏi!" 

    if not schema.strip():
        return "Vui lòng cung cấp database schema!"

    #tạo đầu vào
    input_text = build_input(question, schema)

    #tokenize 
    inputs = tokenizer(input_text,return_tensors='pt',truncation=True,max_length=640) 

    #Đưa dữ liệu (dạng tensor) lên cùng thiết bị với model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }    

    #sinh SQL 
    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens = 256,
            num_beams=4,
            do_sample=False
        )

        #giải mã kết quả
        sql = tokenizer.decode(outputs[0],skip_special_tokens=True)

        return sql.strip()

#Xây dựng giao diện Gradio

demo = gr.Interface(
    fn = predict_sql,
    inputs=[
        gr.Textbox(
            label='Câu hỏi bằng ngôn ngữ tự nhiên',
            placeholder='How many singers are there',
            lines=3
        ),
        gr.Textbox(
            label='Database Schema',
            placeholder='Nhập schema của cơ sở dữ liệu',
            lines=8
        )
    ],

    outputs = gr.Textbox(
        label='SQL được model dự đoán',
        lines=6
    ),

    title='Text-to-SQL Demo',
    
    description= (
        "Nhập câu hỏi và database schema."
        "Model T5 đã fine-tune sẽ sinh câu lệnh SQL tương ứng. "
    )
)

#5. Khởi chạy web   

if __name__ == "__main__":
    demo.launch()
