from PyPDF2 import PdfFileWriter, PdfFileReader
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

FilesPath = "UPLOADS/"

class WaterMark:
    def add_watermark(input_file, output_file, text):
        print(input_file)
        input_file = open('UPLOADS/'+input_file, 'rb')
        pdf_reader = PdfFileReader(input_file, strict=False)

        output_writer = PdfFileWriter()

        for page_num in range(pdf_reader.numPages):

            page = pdf_reader.getPage(page_num)

            packet = BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)

            # Select a font that is recognized by ReportLab
            font_name = "Helvetica"  # Change this to any font you have installed
            can.setFont(font_name, 10)

            watermark_text = text

            page_width = float(page.mediaBox.getWidth())  # Convert to float
            page_height = float(page.mediaBox.getHeight())  # Convert to float

            x = (page_width - can.stringWidth(watermark_text)) / 2  # Calculate x position
            y = (page_height - can.stringWidth(watermark_text)) / 2  # Calculate y position

            can.rotate(20)
            can.setFillAlpha(0.5)
            can.setFillColorRGB(255, 255, 255)
            can.drawString(x, y, watermark_text)
            can.save()

            packet.seek(0)

            watermark_pdf = PdfFileReader(packet)

            page.mergePage(watermark_pdf.getPage(0))

            output_writer.addPage(page)

        output_path = open(output_file, 'wb')
        output_writer.write(output_path)

        input_file.close()
        output_path.close()