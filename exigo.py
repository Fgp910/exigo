'''
This module implements the ActivationExplainer class, which generates
explanations for a Keras model from its training dataset using how the neurons
of the network activate for each train case (i.e. the activation patterns of
the training set).
'''
import random
import numpy as np
import keras
from keras import backend as K
from scipy.spatial import distance as sp_distance


class ActivationExplainer:
    '''A simple explanation generator based on the activation pattern of the training cases of a Keras model. '''
    def __init__(self, model, x_train, verbose=0):
        '''Initializes the explainer from the target model and its training dataset. It creates an activation pattern
        extractor and stores the activation patterns of the training cases.

        :param model: a Keras model.
        :type model: keras.Model
        :param x_train: the training data.
        :type x_train: numpy.array
        :param verbose: verbosity mode for the activation patterns extraction. 0=silent, 1=verbose.
        :type verbose: int
        '''
        self.model = model
        self.x_train = x_train
        K.clear_session()   # Resets layer names
        flat = keras.layers.Flatten()
        self.extractor = keras.Model(
            inputs=self.model.inputs,
            outputs=[flat(layer.output) for layer in self.model.layers],
        )

        # Activation pattern extraction
        aux = self.extractor.predict(self.x_train, verbose=verbose)
        self.activations = [[layer[i] for layer in aux] for i in range(len(self.x_train))]


    @staticmethod
    def cosine(vec1, vec2):
        '''Computes the cosine distance between two vectors, which is equal to 1 minus the cosine of the angle they
        form.

        :param vec1: The first vector to be compared.
        :type vec1: list or numpy array
        :param vec2: The second vector to be compared.
        :type vec2: list or numpy array

        :return: The cosine distance between vec1 and vec2.
        :rtype: float
        '''
        if np.array_equal(vec1, vec2):
            return 0
        return sp_distance.cosine(vec1, vec2) if (vec1.any() and vec2.any()) else np.inf


    def __get_distances(self, new_activation, distance_function, weights=None):
        '''Returns a numpy array with the distances between a new activation pattern and each training activation
        pattern. The distance between two activation patterns is the weighted sum of the distance (given by
        distance_function) between each layer:
            distance(activation1, activation2) = avg(distance_function(activation1.L, activation2.L) for L in layers)
        '''
        distances = []
        for train_activation in self.activations:
            case_distances = np.array([distance_function(new_activation[i], train_activation[i])
                                       for i in range(len(self.model.layers))])
            distances.append(np.average(case_distances, weights=weights))

        return np.array(distances)


    def __get_similarities(self, new_activation, distance_function, weights=None, alpha=1, beta=3):
        '''Returns a numpy array with the similarities between a new activation pattern and each training activation
        patern. The similarity between activation patterns is 2^{-(d/alpha)^beta} where d is the distance between the
        patterns, alpha is the distance fixed as 0.5 similarity and beta is a parameter that controls the curvature of
        the similarity function: bigger values of beta make the function map distances less than alpha to similarities
        much closer to 1, behaving in the limit as a step function on alpha.
        '''
        distances = self.__get_distances(new_activation, distance_function, weights=weights)
        return np.exp2(-(distances/alpha)**beta)


    def __get_aproximate_diameter(self, distance_function, weights=None):
        '''Returns an approximation of the training activations' diameter'''
        reference = random.choice(self.activations)

        distances = self.__get_distances(reference, distance_function, weights=weights)
        return distances.max()


    def explain(self, new_case, distance_function, weights=None, top_k=None, threshold=None, beta=3):
        '''Returns training cases whose activation values are most similar to those of the new case presented.

        :param new_case: the target of the explanation. It must be a numpy array of the shape of the model's input.
        :type new_case: numpy.ndarray
        :param distance_function: the function used to compare the activation values of each pair of layers. It must
            return positive values and must be 0 when comparing a layer's activation vector with itself.
        :type distance_function: callable
        :param weights: an array of weights for the distances of each pair of layers, used to calculate the final
            similarity value between cases. If set to None, each layer will be given the same weight in the last
            weighted sum.
        :type weights: list
        :param top_k: an integer whose value determines how many training cases will be returned, from most to least
            similar. If set to None, all training cases will be returned in order.
        :type top_k: int
        :param threshold: a decimal value between 0 and 1. The function will return only those training cases whose
            similarities are above the threshold. If the parameter top_k is set, threshold will be ignored. If set to
            None, all training cases will be returned in orden.
        :type threshold: float
        :param beta: a decimal value used in the function that transforms distances into similarities. Namely, this
            transformation is given by the formula sim = 2^{-(d/alpha)^beta}, where d is the distance between the
            patterns, alpha is an approximation of the radius of the training cases' set and beta is a parameter that
            controls the curvature of the similarity function: bigger values of beta make the function map distances
            less than alpha to similarities much closer to 1, behaving in the limit as a step function on alpha.
        :type beta: float

        :returns: Two lists with the most similar training cases and their similarity scores, respectively.
        :rtype: tuple(list, list)
        '''

        # new_case activation pattern extraction
        aux = self.extractor.predict(np.array([new_case]), verbose=0)
        new_activation = [layer[0] for layer in aux]

        # Similarities calculation
        approx_radius = self.__get_aproximate_diameter(distance_function, weights=weights)/2.0
        similarities = self.__get_similarities(new_activation, distance_function, weights=weights, alpha=approx_radius,
                                               beta=beta)

        # Pick the indices to return
        if top_k is not None:
            indices = np.argpartition(-similarities, top_k)[:top_k]
        elif threshold is not None:
            indices = [idx[0] for idx in np.argwhere(similarities >= threshold)]
            if len(indices) == 0:
                return [], []
        else:
            indices = np.array(range(len(self.activations)))  # By default, all the indices

        final_similarities = -np.sort(-similarities[indices])
        # Retrieve the indices taking into account that there might be repeated similarities
        final_indices = []
        skip = 0
        for sim in final_similarities:
            if skip > 0:
                skip -= 1
                continue
            aux = np.where(similarities == sim)
            skip = len(aux[0]) - 1
            final_indices = np.append(final_indices, aux)

        return final_indices.astype(int), final_similarities
