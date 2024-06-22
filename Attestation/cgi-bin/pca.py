#!/usr/bin/env python3
import cgi
import cgitb
import os

cgitb.enable()

form = cgi.FieldStorage()
destname = form.getvalue('destname')
fileitem = form['file']

fileitem.file.seek(0, 2)
size = fileitem.file.tell()
fileitem.file.seek(0)
name, extension = os.path.splitext(fileitem.filename)

message = "Failed"
if size < 4096:
    data = fileitem.file.read()
    # this is the base name of the file that was uploaded:
    #filename = os.path.basename(fileitem.filename)
    filename = os.path.join(os.getcwd(), "PrivacyCA", os.path.basename(fileitem.filename))
    if destname != ".":
        filename = os.path.join(os.getcwd(), "PrivacyCA", destname)
    with open(filename, 'wb') as f:
        f.write(data)
    message = "Success"
    
#print('Content-Type: text/plain\r\n\r\n', end='')
#print(message)
