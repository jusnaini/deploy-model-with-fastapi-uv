from fastapi import FastAPI
import uvicorn as uv

app = FastAPI(title="ping")

@app.get("/ping")
def ping():
    # return "PONG"
    # return {'message':'Pong..Hello World!'}
    return 'Pong..Hello World!'

if __name__ == "__main__" :
    uv.run(app,host="0.0.0.0", port=9696)
