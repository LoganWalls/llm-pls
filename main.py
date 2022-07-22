from llm_pls import app, config

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("llm_pls.app:app", host="0.0.0.0", port=config.port, reload=config.debug)
