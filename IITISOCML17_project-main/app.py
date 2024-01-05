from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import subprocess
import uvicorn

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def main():
    return HTMLResponse(content=open("static/index.html").read(), status_code=200)

@app.get("/Contacts")
async def Contacts():
    return HTMLResponse(content=open("static/Contacts.html").read(), status_code=200)

@app.post("/submit")
async def get_info(image: UploadFile = File(...), message: str = Form(...)):
    try:
        # Save the image to a directory or perform any other operations you want
         
        msg_str = message
        contents = await image.read()
        img_path = f"static/{image.filename}"
        with open(img_path, "wb") as f:
            f.write(contents)    

        subprocess.check_call(['python', 'generate.py', '--image', img_path, '--message', msg_str], bufsize=0 )
       
        
        return HTMLResponse(content=open("static/output.html").read(), status_code=200)
        
         
        
    except Exception as e:
        return HTMLResponse(content=open("static/error.html").read(), status_code=500)
    
@app.get("/download")
async def download_output(filename: str):
    file_path = Path(filename)
    if file_path.exists():
        return FileResponse(file_path, headers={"Content-Disposition": f"attachment; filename={file_path.name}"})
    else:
        return HTMLResponse(content=open("static/error.html").read(), status_code=404)
    
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
     


