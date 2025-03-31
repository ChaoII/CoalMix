from main import app
import uvicorn
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the application with a specified parameters.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--port", type=int, default=5553, help="Port to run the application on (default: 5053)")
    parser.add_argument("--workers", type=int, default=1, help="workers for the server")
    args = parser.parse_args()
    uvicorn.run(app=app, host=args.host, port=args.port, workers=args.workers)
