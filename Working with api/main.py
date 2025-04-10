import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "qwen/qwen2.5-vl-32b-instruct:free",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": """Analyze the provided image or video frame for potential exam malpractice. In an exam setting, students are expected to work silently, independently, and avoid any form of communication or interaction with others. Carefully check for any of the following indicators of cheating behavior:

Students turning their heads sideways or backward to look at another student's paper.

Students making eye contact or any kind of hand gestures to communicate with each other.

Passing or receiving slips of paper, notes, or any unauthorized materials.

Using mobile phones, earphones, or any electronic devices.

Unusual body posture suggesting attempts to view others’ answer sheets.

Looking away from their own paper for prolonged periods, especially toward another student.

Based on this criteria, does the image show any suspicious behavior or signs of cheating? Provide a detailed analysis of the scene, including the posture, gaze direction, hand positions, and any other relevant visual cues"""
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://cheatingdetectiontesting1.s3.us-east-1.amazonaws.com/test.jpeg"
            }
          }
        ]
      }
    ],
    
  })

)

print(response.json())
print("\n")
print("\n")
print("\n")
print(response.json()["choices"][0]["message"]["content"])