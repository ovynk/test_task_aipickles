# Setup
Run on Python 3.10

1. Create a virtual environment:
```
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate.bat`
```
2. Install dependencies:
```
    pip install -r requirements.txt
```
3. Set your huggingface token in main.py.  Edit Access Token Permissions, check the box 'Make calls to the serverless Inference API'
   in Inference and 'Write access to contents/settings of all repos under your personal namespace' in Repos.
5. Run the application:
```
    uvicorn main:app --reload
```
5. Send a POST request to http://localhost:8000/summarize with a JSON body containing the text to be summarized.

Example
![example_json_body](https://github.com/ovynk/test_task_aipickles/assets/90598021/d5797314-6240-4cc7-a4f9-60d5435b6271)
