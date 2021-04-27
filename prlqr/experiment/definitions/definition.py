class Settings:

    def __init__(self, **settings):
        self.__dict__.update(**settings)

    def to_dict(self):
        data = self.__dict__.copy()
        data['system'] = data['system'].to_dict()

        if 'training_data' in data:
            data['training_data'] = data['training_data'].copy()
            data['training_data']['controller'] = data['training_data']['controller'].to_dict()

        return data