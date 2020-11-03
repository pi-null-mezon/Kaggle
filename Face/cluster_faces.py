# ------------------------------------------------------------------
# Designed to cluster photos by same face
# Input - directory filled by photos of human faces
# Output - directory with subdirs, one subdir for each unique person
# ------------------------------------------------------------------

import requests
import os
import json
from shutil import copyfile


same_person_distance_threshold = 0.55

inputdirname = '/home/alex/Downloads/photo'

outputdirname = inputdirname + '_clustered'

if not os.path.exists(outputdirname):
    os.mkdir(outputdirname)

person = 0

for item in os.listdir(inputdirname):
    if '.jpg' in item or '.jpeg' in item or '.png' in item:
        print(f"UNDER PROCESSING: {item}")
        if person == 0:
            label = str(f"person_{person:04}")
            labeldir = os.path.join(outputdirname, label)
            os.mkdir(labeldir)
            copyfile(os.path.join(inputdirname, item), os.path.join(labeldir, item))
            result = requests.request('POST', 'http://localhost:5000/iface/remember',
                                      data={'labelinfo': label},
                                      files={'file': (item, open(os.path.join(inputdirname, item), 'rb'),
                                                      'image/png' if '.png' in item else 'image/jpeg')})
            print(f"  R - {item:15s} - [{result.status_code}]")
            person += 1
        else:
            result = requests.request('POST', 'http://localhost:5000/iface/identify',
                                      files={'file': open(os.path.join(inputdirname, item), 'rb')})
            reply = json.loads(result.text)
            print(f"  I - {item:15s} - [{result.status_code}] - {reply}")
            if result.status_code == 200:
                if reply['distance'] > same_person_distance_threshold:
                    label = str(f"person_{person:04}")
                    labeldir = os.path.join(outputdirname, label)
                    os.mkdir(labeldir)
                    copyfile(os.path.join(inputdirname,item), os.path.join(labeldir, item))
                    result = requests.request('POST', 'http://localhost:5000/iface/remember',
                                              data={'labelinfo': label},
                                              files={'file': (item, open(os.path.join(inputdirname, item), 'rb'),
                                                              'image/png' if '.png' in item else 'image/jpeg')})
                    print(f"  R - {item:15s} - [{result.status_code}]")
                    person += 1
                else:
                    label = reply['labelinfo']
                    labeldir = os.path.join(outputdirname, label)
                    copyfile(os.path.join(inputdirname, item), os.path.join(labeldir, item))

