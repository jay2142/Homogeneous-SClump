import json
import numpy as np
from pysclump import PathSim, SClump

# I scan a limited number of businesses and users so that the program completes quickly.
# To my understanding, there is no notable pre-existing ordering in the datasets, so scanning N entries from the top should be fine.
NUM_BUSINESSES = 2001
NUM_USERS = 5001

# These are just used to improve time complexity (I need to quickly check if I've already seen a city/business/etc)
# I should actually use dictionaries instead, but this suffices for the time being
businesses_set = set() 
cities_set = set()
users_set = set()
categories_set = set()

businesses = []
cities = []
users = []
categories = []

type_lists = {
    'B': businesses,
    'C': cities,
    'U': users,
    'A': categories
}

# 160585 businesses
# 836 cities
# 1330 categories

BC = np.zeros((NUM_BUSINESSES, 836))
BA = np.zeros((NUM_BUSINESSES, 1330))
BU = np.zeros((NUM_BUSINESSES, NUM_USERS))

with open('data/yelp_academic_dataset_business.json') as f:
    n = 0
    for line in f:
        if n > NUM_BUSINESSES-1:
            break
        n += 1

        data = json.loads(line)
        id = data['business_id']
        # state = data['state']
        city = data['city']
        if data['categories']:
            category_list = data['categories'].split(', ')
        else:
            category_list = []

        if id not in businesses_set:
            businesses_set.add(id)
            businesses.append(id)
        if city not in cities_set:
            cities_set.add(city)
            cities.append(city)
        for category in category_list:
            if category not in categories_set:
                categories_set.add(category)
                categories.append(category)
        
        business_idx = len(businesses)-1
        city_idx = cities.index(city)
        BC[business_idx, city_idx] += 1
        for category in category_list:
            category_idx = categories.index(category)
            BA[business_idx, category_idx] += 1

with open('data/yelp_academic_dataset_review.json') as f:
    n = 0
    for line in f:
        if n > NUM_USERS-1:
            break
        
        data = json.loads(line)
        
        business_id = data['business_id']
        if business_id not in businesses_set:
            continue # Business is unknown
        n += 1

        user_id = data['user_id']
        if user_id not in users_set:
            users_set.add(user_id)
            users.append(user_id)
        
        
        business_idx = businesses.index(business_id)
        user_idx = users.index(user_id)
        BU[business_idx, user_idx] += 1


incidence_matrices = {
    'BC': BC,
    'BA': BA,
    'BU': BU
}
        
ps = PathSim(type_lists, incidence_matrices)

similarity_matrices = {
    'BCB': ps.compute_similarity_matrix(metapath='BCB'),
    'BUB': ps.compute_similarity_matrix(metapath='BUB'),
    'BAB': ps.compute_similarity_matrix(metapath='BAB')
}

print("similarity matrices constructed")
sclump = SClump(similarity_matrices, num_clusters=201)
print("sclump set up")
labels, learned_similarity_matrix, metapath_weights = sclump.run()
print(labels)
print(learned_similarity_matrix)
print(metapath_weights)