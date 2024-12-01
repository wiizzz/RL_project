from openai import OpenAI
import os
import json

import base64
import requests

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


base_path = "hud/imgs"


client = OpenAI(api_key = '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')

# response = client.chat.completions.create(
#   model="gpt-4-vision-preview",
#   temperature=1.2,
#   messages=[
#     {
#       "role": "user",
      
#       "content": [
#         {"type": "text", "text": "This image is a screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the specific situation shown in the image, please give me a detailed description of the environment the agent is seeing now, and please focus more on describing the wall’s and obstacle’s position and orientation, also please use the o’clock notation to specify the directions."
# },
#         # Please tell me how many enemies are there in the agents view.
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": f"data:image/jpeg;base64,{base64_image1}",
#           },
#         },
#       ],
#     }
#   ],
#   max_tokens=1000,
# )

# print(response.choices[0])
# print(response.choices[0].message.content)

image_num = 100
idx = 0 
all_descriptions = []

for i in range(idx,idx+image_num):
  image_path = os.path.join(base_path,f"{i}.jpg")
  base64_image = encode_image(image_path)
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    temperature=1.2,
    messages=[
      {
        "role": "user",
        
        "content": [
          {"type": "text", "text": "This image is a screen of a gameplay in the first person shooting game 'DOOM’, containing information of its location and the environment and the enemies. Based on the specific situation shown in the image, please give me a detailed description of the environment the agent is seeing now (no need to describe the environment behind the agent), and please focus more on describing 1.the wall’s and obstacle’s position and orientation 2.enemy's position(if there is any) 3.item's position(please list them all if there is any, and there are 2 kinds of items available : (a)health package, a white box chararcterized with a bit of red on its right and green on its left, (b)ammo, a brown colored shell that is relatively smaller in size ), also please use the the angle notation of left or right theta degrees to the agent to specify the directions."
  },
          # Please tell me how many enemies are there in the agents view.
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
            },
          },
        ],
      }
    ],
    max_tokens=1000,
  )

  all_descriptions.append(
    {
      #"response_object" :response.choices[0],
      "idx" : idx,
      "content" : response.choices[0].message.content
    }
  )

  idx += 1

  # print(response.choices[0])
  print(response.choices[0].message.content)



with open(f"first_prompt_content.json", "+w", encoding="utf-8") as f:
    json.dump(all_descriptions, f, indent=4, ensure_ascii=False)