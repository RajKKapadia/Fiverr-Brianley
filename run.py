if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app="src.main:app", port=os.getenv("PORT", 8080), reload=True)
