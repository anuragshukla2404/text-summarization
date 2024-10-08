from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.conponents.model_trainer import ModelTrainer
from src.textSummarizer.logging import logger

class ModelTrainerPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()