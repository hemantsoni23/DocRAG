from frontend.interface import create_demo
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False) 