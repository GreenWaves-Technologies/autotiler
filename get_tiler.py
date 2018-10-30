import requests
import json
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def fatal_error(error):
    print (bcolors.FAIL + 'FATAL ERROR: ' + str(error) + bcolors.ENDC)
    sys.exit(1)

def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True, timeout=3)
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
    return local_filename

ri = vars(__builtins__).get('raw_input',input)
print("Registration is required to load the GAP8 AutoTiler library\n")
print("You will be prompted for your name, company and email address and the")
print("link for the AutoTiler libray will be sent to your email address.")
print("This information is used uniquely to keep track of AutoTiler users.")
name = ri("Enter your name: ")
company = ri("Enter your company name: ")
email = ri("Enter your email address: ")

url = 'https://hooks.zapier.com/hooks/catch/2624512/e6qico/'
payload = { 'name': name, 'company': company, 'email': email }
headers = { 'content-type': 'application/json' }

print("Triggering email ... please wait")
try:
    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=3)
    response.raise_for_status()
except requests.exceptions.RequestException as ex:
    fatal_error(ex)


print("Please check your email and copy and paste the URL in the email below")
url = ri("Enter URL from email: ")
try:
    f = open(".tiler_url","w+")
    f.write(url)
    f.close()
except:
    fatal_error("problem writing file")