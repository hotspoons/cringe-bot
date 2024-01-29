import os
import re
import sys
import json
from html2text import HTML2Text

html_converter = HTML2Text()
html_converter.unicode_snob = 1
html_converter.ignore_emphasis = 1
html_converter.ignore_images = 1
html_converter.ignore_links = 1
html_converter.body_width = 0

USER='user'
ASSISTANT='assistant'

def convert_htmls(path, screennames):
    output = ""
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".htm") or file.endswith(".html"):
                with open(os.path.join(root, file), "r") as f:
                    output += filter(html_converter.handle(f.read()), screennames)
    return output

def to_convo_format(input):
    turns = 0
    id = 0
    last_role = ""
    list_of_convos = []
    convo_struct = {
        'user': '',
        'assistant': '',
        'id': str(id)
    }
    list_of_convos.append(convo_struct)
    for line in input.split("\n"):
        if line.startswith('assistant :'):
            role = ASSISTANT
        elif line.startswith('user :'):
            role = USER
        else:
            role = USER if last_role == "" else last_role
            
        if last_role == "": #init
            last_role = role
        if turns == 1 and last_role != role:
            id += 1
            convo_struct = {
                'user': '',
                'assistant': '',
                'id': str(id)
            }
            list_of_convos.append(convo_struct)
            turns = 0
        elif last_role != role:
            turns += 1
        if convo_struct[role] != '':
            convo_struct[role] += '\n'
        convo_struct[role] += line.replace(f"{role} :", '').strip()
        last_role = role
    return list_of_convos

def filter(text, screennames):
    lines = text.splitlines()
    screennames = screennames.split(",")
    filtered = []
    for line in lines:
        if "is idle at " in line or "is no longer idle at " in line or \
            " returned at " in line or "is away at " in line or \
            "Session concluded at " in line or \
            "Auto response from " in line or "* * *" in line or \
            "wants to directly connect " in line or \
            "is now directly connected " in line or \
            "Your screen name " in line or \
            "direct connection is closed " in line or \
            " signed off at " in line or \
            " signed on at " in line :
            continue
        line = re.sub(r'\((\d{1,2}:\d{2}:\d{2}\s*(AM|PM))\)', '', line)
        for screenname in screennames:
            line = line.replace(f"{screenname} :", "assistant___")
        line = re.sub(r"^([^:]*):", 'user :', line, 0, re.MULTILINE)
        line = line.replace("assistant___", "assistant :")
        if line.isspace() == True or line == "":
            continue
        filtered.append(line)
    return "\n".join(filtered)

# arg 1 is the path to where you wish to scan for AIM chat logs,
# arg 2 is a comma separated list of AIM screennames to use as the assistant role
if __name__ == "__main__":
   if len(sys.argv) < 3:
       print("Usage: python aim_log_scraper /path/to/aim/chats MyFirstScreenname,MySecondScreenname")
       exit(0)
   output = convert_htmls(sys.argv[1], sys.argv[2])
   convos = to_convo_format(output)
   for convo in convos:
       print(json.dumps(convo))
