from pyspark import SparkContext
from datetime import datetime, timedelta
import csv
import functools
import json
import numpy as np
import sys
 
def main(sc):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    rddPlaces = sc.textFile('hdfs:///data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*')
    OUTPUT_PREFIX = sys.argv[1]
    
    CAT_CODES = {'445210', '445110', '722410', '452311', '722513', '445120', '446110', '445299', '722515', '311811', '722511', '445230', '446191', '445291', '445220', '452210', '445292'}
    CAT_GROUP = {'452210': 0, '452311': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446110': 5, '446191': 5, '722515': 6, '311811': 6, '445210': 7, '445299': 7, '445230': 7, '445291': 7, '445220': 7, '445292': 7, '445110': 8}

    def filterPOIs(_, lines):
        for line in lines:
            line = line.split(',')
            if line[9] in CAT_CODES:
                yield (line[0], CAT_GROUP[line[9]])
        
    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs) \
            .cache()

    storeGroup = dict(rddD.collect())
    groupCount = rddD \
        .map(lambda x: (x[1], 1)) \
        .reduceByKey(lambda x,y: x+y) \
        .sortBy(lambda x: x) \
        .map(lambda x: x[1]) \
        .collect()

    def extractVisits(storeGroup, _, lines):
        for line in lines:
            line = next(csv.reader([line]))
            if line[0] in storeGroup:
                dates = [str(datetime.strptime(line[12][:10],'%Y-%m-%d') + timedelta(days=i))[:10] for i in range(7)]
                for i, date in enumerate(dates):
                    if date[:4] in ['2019', '2020']:
                        date = (datetime.strptime(date,'%Y-%m-%d') - datetime.strptime('2019-01-01', '%Y-%m-%d')).days
                        visit_of_the_day = line[16][1:-1].split(',')
                        yield ((storeGroup[line[0]], date),  int(visit_of_the_day[i]))
    

    rddG = rddPattern \
        .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))

    def computeStats(groupCount, _, records):
        for record in records:
            key, value = record[0], list(record[1])
            if len(value) < groupCount[key[0]]:
                padding = [0]*(groupCount[key[0]] - len(value))
                value.extend(padding)
            median = int(np.median(value))
            stdv = np.std(value)
            low = 0 if (median-stdv) <0 else int(median-stdv)
            high = int(median+stdv)
            yield (key, (median, low, high))
    

    rddH = rddG.groupByKey() \
            .mapPartitionsWithIndex(functools.partial(computeStats, groupCount))

    def padding(record):
        key, value = record[0], list(record[1])
        if len(value) < groupCount[key[0]]:
            padding = [0]*(groupCount[key[0]] - len(value))
            value.extend(padding)
        return key, value

    rddI = rddG.groupByKey() \
            .map(lambda x: padding(x)) \
            .map(lambda x: (x[0], np.median(list(x[1])), np.std(list(x[1])) )) \
            .map(lambda x: (x[0][0], x[0][1], int(x[1]), int(x[1]- x[2]), int(x[1]+x[2]) )) \
            .map(lambda x: (x[0], str(datetime.strptime('2019-01-01', '%Y-%m-%d')+timedelta(days = x[1]))[:10], x[2], x[3]if x[3]>0 else 0, x[4])) \
            .map(lambda x: (x[0], x[1][:4], '2020-'+x[1][5:], x[2], x[3], x[4])) \
            .map(lambda x: (x[0], ','.join([str(cell) for cell in x[1:]])))

    rddJ = rddI.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    #rddJ = (header + rddJ).coalesce(10).cache()


    filenames = ['big_box_grocers',
    'convenience_stores',
    'drinking_places',
    'full_service_restaurants',
    'limited_service_restaurants',
    'pharmacies_and_drug_stores',
    'snack_and_bakeries',
    'specialty_food_stores',
    'supermarkets_except_convenience_stores']
    for i, name in enumerate(filenames):
        rddJ = (header + rddJ).coalesce(10).cache()
        rddJ.filter(lambda x: x[0]==i or x[0]==-(i+1)).values() \
            .saveAsTextFile(f'{OUTPUT_PREFIX}/{name}')

 
if __name__=='__main__':
    sc = SparkContext()
    main(sc)