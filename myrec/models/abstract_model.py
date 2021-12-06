class AbstractModel:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    # dataset的input结构要和model的InputLayer格式对应，因此可以根据input结构进一步抽象dataset
    def build_input_layer(self):
        pass

    def get_model(self):
        pass
