from openai import OpenAI
import re
import base64
import requests
import json
import os


import base64

# Specify the path to your JSON file
json_file_path = 'first_prompt_content.json'

# Read JSON data from the file
with open(json_file_path, 'r') as file:
    data = json.load(file)

if data:
    first_dictionary = data[0]

def pars(input_string):
  #input_string = "[3(0), 2, 1(30), 4, 2]"

  # Define regular expressions for values with and without parentheses
  value_pattern = re.compile(r'(\d+)(?:\((\d+)\))?')

  # Find all matches in the input string
  matches = re.findall(value_pattern, input_string)

  # Initialize variables
  values_without_parenthesis = []
  values_with_parenthesis = []

  # Extract values from matches
  for match in matches:
    value_without_parenthesis = int(match[0])
    value_with_parenthesis = int(match[1]) if match[1] != '' else 0

    values_without_parenthesis.append(value_without_parenthesis)
    values_with_parenthesis.append(value_with_parenthesis)

  # Print the results
  # print("Values without parenthesis:", values_without_parenthesis)
  # print("Values with parenthesis:", values_with_parenthesis)

  # Assign values to individual variables (optional)
  var1, var2, var3, var4, var5 = values_without_parenthesis[:5]
  var6, var7, var8, var9, var10 = values_with_parenthesis[:5]

  variable_names = [f"var{i}" for i in range(1, 11)]

  actions = []
  for i in range(10):
      actions.append(locals()[variable_names[i]])
  
  # Print individual variables (optional)
  print("Individual variables without parenthesis:", var1, var2, var3, var4, var5)
  print("Individual variables with parenthesis:", var6, var7, var8, var9, var10)

  return actions




p1 = "The below text enclosed in brackets { } is the description of the screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies.{ "
p2 = "} Based on the specific situation described, please imagine you are a pro player and form a plan of 5 consecutive actions listed below to do at this moment so that you can optimize in navigating through the map and eliminating enemies. Please perceive that all enemies are able to deal long range attacks like shooting, so you need to consider the risk when collecting a health package in front of an enemy. Please output the number (an integer) and its direction (in degree format, let 90 degrees to the right be 0 and 90 degrees to the  left be 180) that represents the action only, and further explanation is not needed to be included in your output. For example, the output should be like : [2(30),1,2,3,5]. Below are the actions you can choose from : 1.Aim and shoot  2.Take cover behind the wall or obstacle in direction of 0, 30, 60, 90, 120, 150, 180 degrees (pick from this 7 directions) 3.Collect health package within sight 4.Collect ammo within sight 5.navigate around the map"


client = OpenAI(api_key = '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')

image_num = 100
data = data[0:image_num]
idx = 0 
all_actions = []

for dic in data:
  response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    temperature=1.2,
    messages=[
      {
        "role": "user",
        
        "content": [
          {"type": "text", "text":p1 + dic["content"].rstrip('\n') + p2
          },
          # Please tell me how many enemies are there in the agents view.
          # {
          #   # "type": "image_url",
          #   # "image_url": {
          #   #   "url": f"data:image/jpeg;base64,{base64_image1}",
          #   # },
          # },
        ],
      }
    ],
    max_tokens=1000,
  )
  r = response.choices[0].message.content
  print(r)
  try :
    p =pars('['+r+']') 
  except : 
    p = 0
  all_actions.append(
    {
      #"response_object" :response.choices[0],
      "idx" : idx,
      "content" : response.choices[0].message.content,
      'action_list': p
    }
  )

  idx += 1

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo-1106",
#   temperature=1.2,
#   messages=[
#     {
#       "role": "user",
      
#       "content": [
#         {"type": "text", "text":p1 + first_dictionary["content"].rstrip('\n') + p2
#         },
#         # Please tell me how many enemies are there in the agents view.
#         # {
#         #   # "type": "image_url",
#         #   # "image_url": {
#         #   #   "url": f"data:image/jpeg;base64,{base64_image1}",
#         #   # },
#         # },
#       ],
#     }
#   ],
#   max_tokens=1000,
# )


# r1 = response.choices[0].message.content
# print(r1)
# print(pars('['+r1+']'))
with open(f"second_prompt_action_all.json", "+w", encoding="utf-8") as f:
    json.dump(all_actions, f, indent=4, ensure_ascii=False)

