from reportlab.pdfgen import canvas

# create a new PDF file
pdf_file = canvas.Canvas('example.pdf')

# add some text
pdf_file.drawString(100, 750, "LEONTEQSECURITY")

with open("./audit_trail.log", 'r') as f:
    lines = f.readlines()
    for line in lines:
        pdf_file.drawString(
            100, 700, line)


# save the PDF file
pdf_file.save()
