from collections import Counter
from math import log
from random import randint

# region Global variables
train_file_path = "train.txt"
test_file_path = "test.txt"
output_file_path = "output.txt"
attributes = []
attributes_to_types = {}
train_samples = []
test_samples = []
classification_attribute = ""
ALG_NAIVE_BAYES = "naive bayes"
ALG_DEC_TREE = "decision tree"
ALG_KNN = "knn"
TAB = '\t'
NEWLINE = '\n'
# endregion

# region Common methods

def get_data_from_file(file_path, get_attributes = False):
        samples_as_dicts_list = []
        with open(file_path, 'r') as samples_file:
                attributes = samples_file.readline().strip(NEWLINE).split(TAB)
                samples = samples_file.read().split(NEWLINE)
        for sample in samples:
                attributes_dict = dict([])
                sample_values = sample.split(TAB)
                for index in range(len(sample_values)):
                        attributes_dict[attributes[index]] = sample_values[index]
                if (len(attributes_dict) == len(attributes)):        
                        samples_as_dicts_list.append(attributes_dict)
        if (get_attributes):
                for att in attributes:
                        types = set()
                        for sample in samples_as_dicts_list:
                                types.add(sample[att])
                        attributes_to_types[att] = list(types)
                return samples_as_dicts_list, attributes, attributes[-1]      
        return samples_as_dicts_list

def calculate_accuracies(prediction_lists):
        result = []
        actual_classifications = [sample[classification_attribute] for sample in test_samples]
        num_of_samples = len(actual_classifications)
        for list in prediction_lists:
                good_predictions = sum([1 for index, item in enumerate(list) if item == actual_classifications[index]])
                result.append(round(float(good_predictions) / num_of_samples, 2))  
        return result

def find_most_frequent_classification(samples_list):
        samples_classification = [sample[classification_attribute] for sample in samples_list]
        counter = Counter(samples_classification)
        return counter.most_common(1)[0][0] # return the most common classification from this tuple

# endregion

# region KNN algorithm
# region Helping Methods
def get_distance_between_samples(current_sample, sample):
        distance = sum([1 for attribute in attributes[:-1] if current_sample[attribute] != sample[attribute]])
        return distance

# endregion Helping Methods

def knn_algorithm(current_sample, k = 5):
        if not (len(current_sample) == len(attributes)):
                raise AttributeError("The test sample should have the same number of attributes as the training sample")
        distance_list = []
        for sample in train_samples:
                current_distance = get_distance_between_samples(current_sample, sample)
                sample_distance_tuple = (sample, current_distance)
                distance_list.append(sample_distance_tuple)
        distance_list.sort(key=lambda tuple : tuple[1]) #sort by distance
        k_closest_neighbours = [sample_distance_tuple[0] for sample_distance_tuple in distance_list[:k]] #get the relevant samples
        prediction = find_most_frequent_classification(k_closest_neighbours)
        return prediction

# endregion

#region Naive Bayes algorithm

#region Helping Methods
def multiply_list(list):
        result = 1
        for item in list:
                result = result * item
        return result

def get_conditional_probability(test_value, attribute, current_class):
        k = len(attributes_to_types[attribute])
        num_both_happen = float(sum([1 for sample in train_samples if (sample[attribute] == test_value and sample[classification_attribute] == current_class)]))
        num_class_happens = float(sum([1 for sample in train_samples if sample[classification_attribute] == current_class]))
        conditional_probability = (num_both_happen + 1) / num_class_happens + k
        return conditional_probability


def dumb_rule(possible_classifications):
        positive_classifications = ["yes", "true", "positive", "skynet"]
        for option in positive_classifications:
                if option.lower() in positive_classifications:
                        return option
        return positive_classifications[0]

# endregion Helping Methods

def naive_bayes_algorithm(current_sample):
        possible_classifications = list(set([sample[classification_attribute] for sample in train_samples]))

        # create dictionary of dictionaries of probabilities
        attributes_probabilities = dict() # keys = attributes, values = probabilities of classification
        for attribute in current_sample:
                if attribute == classification_attribute: continue
                single_attribute_prob = dict()
                for option in possible_classifications:
                        single_attribute_prob[option] = get_conditional_probability(current_sample[attribute], attribute, option)
                attributes_probabilities[attribute] = single_attribute_prob
        
        # get a list containing holding a tuple (option, P(option)) for every classification option
        all_class_probs = []
        for option in possible_classifications:
                prob = multiply_list([attributes_probabilities[att][option] for att in attributes_probabilities])
                option_prob_tuple = (option, prob)
                all_class_probs.append(option_prob_tuple)

        #sort by probability
        all_class_probs.sort(key=lambda tuple : tuple[1])

        # get the tuple (most_probable, P(most_probable))
        most_probable_tuple = all_class_probs[-1] 
        final_classification = most_probable_tuple[0]

        if (len(set([class_prob_tuple[1] for class_prob_tuple in all_class_probs])) == 1 ): # check if all prob are equal
                final_classification = dumb_rule(possible_classifications)                

        return final_classification

# endregion

# region Decision Tree algorithm
def create_tree(self, attributes):
        return DecisionTree(attributes)

# region Helping Methods



# endregion

class DecisionTree:
    def __init__(self, attributes, examples):



    def predict(self, case):
            """
            not implemented, returns random value
            """
            value = randint(0 , len(attributes_to_types[classification_attribute]) - 1)
            return attributes_to_types[classification_attribute][value]


# endregion Decision Tree algorithm

if __name__ == "__main__":
        train_samples, attributes, classification_attribute = get_data_from_file(train_file_path, get_attributes = True)
        test_samples = get_data_from_file(test_file_path)

        # DIDN'T HAVE TIME TO COMPLETE THE DECISION TREE ALGORITHM
        # create the decision tree
        samples = [list(sample.values()) for sample in train_samples]
        tree = create_tree(attributes, samples)
        # write_tree_to_file(tree, 'output_tree.txt')

        predictions = {ALG_DEC_TREE:[], ALG_KNN:[], ALG_NAIVE_BAYES:[]}
        lines_list = ["Num\tDT\tKNN\tnaiveBase"]
        row_num = 1
        for sample in test_samples:
                dt = tree.predict(sample)
                knn = knn_algorithm(sample)
                nb = naive_bayes_algorithm(sample)

                predictions[ALG_DEC_TREE].append(dt)
                predictions[ALG_KNN].append(knn)
                predictions[ALG_NAIVE_BAYES].append(nb)
                
                line = TAB.join([str(row_num), dt, knn, nb])
                lines_list.append(line)
                row_num += 1
        # get accuracies
        accuracies = calculate_accuracies([predictions[ALG_DEC_TREE], predictions[ALG_KNN], predictions[ALG_NAIVE_BAYES]])
        lines_list.append(TAB + TAB.join([str(acc) for acc in accuracies]))
        table_as_string = NEWLINE.join(lines_list)

        with open(output_file_path, 'w') as output_file:
                output_file.write(table_as_string)