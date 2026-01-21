from fpdf import FPDF
import os

def export_to_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    
    # Настройка шрифтов
    font_path = "DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font('DejaVu', '', 12)
    else:
        pdf.set_font('Arial', '', 12)

    # Шапка
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(0, 10, txt="ВЕТЕРИНАРНЫЙ ПРОТОКОЛ ОБСЛЕДОВАНИЯ VetAI", ln=True, align='C', fill=True)
    pdf.ln(10)

    # Вспомогательная функция для секций
    def add_pdf_section(title, content_dict):
        pdf.set_font('DejaVu', '', 12) if os.path.exists(font_path) else pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, txt=title, ln=True, border='B')
        pdf.set_font('DejaVu', '', 10) if os.path.exists(font_path) else pdf.set_font('Arial', '', 10)
        pdf.ln(2)
        for key, value in content_dict.items():
            pdf.multi_cell(0, 6, txt=f"{key}: {value}")
        pdf.ln(5)

    # 1. Диагнозы
    add_pdf_section("РЕЗУЛЬТАТЫ АНАЛИЗА", {
        "Основной диагноз": data.get('diag1', 'Н/Д'),
        "Вероятность": data.get('prob1', '0%'),
        "Дифференциальный": data.get('diag2', 'Нет')
    })

    # 2. Анамнез
    add_pdf_section("АНАМНЕЗ И ИСТОРИЯ", {
        "Вид / Возраст": f"{data.get('breed')} / {data.get('age')}",
        "Вакцинация": data.get('vax', 'Н/Д'),
        "Профилактика": data.get('preventive', 'Н/Д')
    })

    # 3. Клинический осмотр
    add_pdf_section("КЛИНИЧЕСКИЕ ДАННЫЕ", {
        "Общее состояние": data.get('status', 'Н/Д'),
        "Вес / Температура": f"{data.get('weight')} / {data.get('temp')}",
        "Сердце / Пульс": f"{data.get('heart')} / {data.get('pulse')}",
        "Дыхание": data.get('resp', 'Н/Д')
    })

    # 4. Симптомы
    pdf.set_font('DejaVu', '', 12) if os.path.exists(font_path) else pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, txt="ВЫБРАННЫЕ СИМПТОМЫ", ln=True, border='B')
    pdf.ln(2)
    pdf.set_font('DejaVu', '', 10) if os.path.exists(font_path) else pdf.set_font('Arial', '', 10)
    symptoms_text = ", ".join(data.get('symptoms', [])) if data.get('symptoms') else "Не указаны"
    pdf.multi_cell(0, 6, txt=symptoms_text)

    return pdf.output(dest='S').encode('latin-1')