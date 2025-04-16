import pandas as pd

class GenderPredictor:
    def __init__(self, dataset_path):
        self.df = pd.read_excel(dataset_path).astype(str)
        self.dataset = self.df.values.tolist()
        self.num = len(self.dataset)
        self.male_count = sum(1 for inst in self.dataset if inst[-1].lower() == "male")
        self.female_count = self.num - self.male_count
        self.prob_male = self.male_count / self.num
        self.prob_female = self.female_count / self.num

    def predict(self, age, placenta, month, belly, history, lifestyle):
        features = [age, placenta.capitalize(), month.capitalize(), belly.capitalize(), history.capitalize(), lifestyle.capitalize()]

        male_likelihood = 1
        female_likelihood = 1

        for i, feature in enumerate(features):
            male_feature_count = sum(1 for inst in self.dataset if inst[i] == feature and inst[-1].lower() == "male")
            female_feature_count = sum(1 for inst in self.dataset if inst[i] == feature and inst[-1].lower() == "female")

            # Laplace smoothing
            male_likelihood *= (male_feature_count + 1) / (self.male_count + 1)
            female_likelihood *= (female_feature_count + 1) / (self.female_count + 1)

        posterior_male = self.prob_male * male_likelihood
        posterior_female = self.prob_female * female_likelihood

        if posterior_male > posterior_female:
            return "Male", posterior_male / (posterior_male + posterior_female)
        else:
            return "Female", posterior_female / (posterior_male + posterior_female)
