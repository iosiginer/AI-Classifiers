from collections import Counter

# region Global variables
train_file_path = "train.txt"
test_file_path = "test.txt"
output_file_path = "output.txt"
attributes = []
attributes_to_num_of_types = {}
train_samples = []
test_samples = []
classification_attribute = ""
# endregion 


def multiply_list(list):
        result = 1
        for item in list:
                result = result * item
        return result


def get_samples_from_file(file_path, get_types = False):
        samples_as_dicts_list = []
        samples = open(file_path, 'r').read().split('\n')
        for sample in samples[1:]:
                attributes_dict = dict([])
                sample_values = sample.split('\t')
                for index in range(len(sample_values)):
                        attributes_dict[attributes[index]] = sample_values[index]
                if (len(attributes_dict) == len(attributes)):        
                        samples_as_dicts_list.append(attributes_dict)
        if (get_types):
                for att in attributes:
                        types = set()
                        for sample in samples_as_dicts_list:
                                types.add(sample[att])
                        attributes_to_num_of_types[att] = len(types)       
        return samples_as_dicts_list


def get_attributes(file_path):
        with open(file_path, 'r') as train_file:
                first_line = train_file.readline()
                attributes = first_line.strip('\n').split('\t')
        return attributes

def get_distance_between_samples(current_sample, sample):
        distance = sum([1 for attribute in attributes[:-1] if current_sample[attribute] != sample[attribute]])
        return distance

def find_most_frequent_classification(k_closest_neighbours):
        neighbours_classification = [sample[classification_attribute] for sample in k_closest_neighbours]
        counter = Counter(neighbours_classification)
        return counter.most_common(1)[0][0] # return the most common classification from this tuple

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



def decision_tree_algorithm(current_sample):
        ## TODO second step in finishing
        return "dt"

def get_conditional_probability(test_value, attribute, current_class):
        num_both_happen = float(sum([1 for sample in train_samples if (sample[attribute] == test_value and sample[classification_attribute] == current_class)]))
        num_class_happens = float(sum([1 for sample in train_samples if sample[classification_attribute] == current_class]))
        conditional_probability = (num_both_happen + 1) / num_class_happens + attributes_to_num_of_types[attribute]
        return conditional_probability


def dumb_rule(possible_classifications):
        positive_classifications = ["yes", "true", "positive", "skynet"]
        for option in positive_classifications:
                if option.lower() in positive_classifications:
                        return option
        return positive_classifications[0]

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

        # if(prob[1] == (1 / len(possible_classifications)) for prob in all_class_probs): #if all probs are equal
        #         final_classification = dumb_rule(possible_classifications)

        return final_classification

def get_all_predictions():
        prediction_lists_dict = dict() 
        dt_predictions = []
        nb_predictions = []
        knn_predictions = []
        lines_list = []
        lines_list.append("Num\tDT\tKNN\tnaiveBase\t\tACTUAL")
        row_num = 1
        for sample in test_samples:
                dt = decision_tree_algorithm(sample)
                dt_predictions.append(dt)

                knn = knn_algorithm(sample)
                knn_predictions.append(knn)
                
                nb = naive_bayes_algorithm(sample)
                nb_predictions.append(nb)
                
                line = '\t'.join([str(row_num), dt, knn, nb])
                # REMOVE THIS LINE, JUST FOR TESTING
                line = line + '\t\t\t\t' + sample[classification_attribute]
                lines_list.append(line)
                row_num += 1
        table_as_string = '\n'.join(lines_list)

        with open(output_file_path, 'w') as output_file:
                output_file.write(table_as_string)

        prediction_lists_dict["KNN"] = knn_predictions
        prediction_lists_dict["NB"] = nb_predictions
        prediction_lists_dict["DT"] = dt_predictions

        return prediction_lists_dict


if __name__ == "__main__":
        attributes = get_attributes(train_file_path)
        train_samples = get_samples_from_file(train_file_path, True)
        test_samples = get_samples_from_file(test_file_path)
        classification_attribute = attributes[-1]

        predictions_dict = get_all_predictions()