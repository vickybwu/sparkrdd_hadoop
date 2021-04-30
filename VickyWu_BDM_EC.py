"""
VickyWu_BDM_HW4.py

Final version
"""

import csv
import numpy as np
import pyspark
import sys
from datetime import datetime, timedelta
sc = pyspark.SparkContext()


maps = {'big_box_grocers': ['452210', '452311'],
        'convenience_stores': ['445120'],
        'drinking_places': ['722410'],
        'full_service_restaurants': ['722511'],    
        'limited_service_restaurants': ['722513'],
        'pharmacies_and_drug_stores': ['446110', '446191'],
        'snack_and_bakeries': ['311811', '722515'],
        'specialty_food_stores': ['445210', '445220', '445230', '445291', '445292',  '445299'],
        'supermarkets_except_convenience_stores': ['445110']
}
names = list(maps.keys())
header = sc.parallelize(['year,date,median,low,high'])

for name in names:
    store_id = set(sc.textFile('hdfs:///data/share/bdm/core-places-nyc.csv') \
        .map(lambda x: x.split(',')) \
        .map(lambda x: (x[1], x[9])) \
        .filter(lambda x: x[1] in maps[name]) \
        .map(lambda x: x[0]) \
        .collect())
    rdd = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*') \
        .map(lambda x: next(csv.reader([x]))) \
        .filter(lambda x: x[1] in store_id) \
        .map(lambda x: (x[12][:10], x[16])) \
        .flatMap(lambda x: [((datetime.strptime(x[0],'%Y-%m-%d') + timedelta(days = i)).date(), [int(w)]) for i, w in enumerate(x[1][1:-1].split(','))]) \
        .reduceByKey(lambda x,y: x+y) \
        .map(lambda x: (x[0], np.median(x[1]), np.std(x[1]) )) \
        .map(lambda x: (str(x[0]), int(x[1]), int(x[1]-x[2]), int(x[1]+x[2]))) \
        .filter(lambda x: x[0][:4] in ['2019', '2020']) \
        .sortBy(lambda x: x) \
        .map(lambda x: (x[0][:4], '2020-'+x[0][5:], x[1], x[2] if x[2]>0 else 0, x[3]if x[3]>0 else 0)) \
        .map(lambda x: ','.join([str(cell) for cell in x]))
    header.union(rdd).saveAsTextFile(sys.argv[-1]+'/'+name) 