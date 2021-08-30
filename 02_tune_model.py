from pyspark.sql import SparkSession

def main(spark):
    '''Main routine for Project
    Parameters
    ----------
    spark : SparkSession object
    '''
    # Load the Data
#     sub1= spark.read.parquet('hdfs:/user/sy1880/subsample1.parquet')
#     sub1.createOrReplaceTempView('sub1')
#     sub5= spark.read.parquet('hdfs:/user/sy1880/subsample5.parquet')
#     sub5.createOrReplaceTempView('sub5')
#     sub25= spark.read.parquet('hdfs:/user/sy1880/subsample25.parquet')
#     sub25.createOrReplaceTempView('sub25')
#     print(sub1.head(),sub5.head(),sub25.head())

    train= spark.read.parquet('hdfs:/user/sy1880/df_train.parquet')
    train.createOrReplaceTempView('train')
    val= spark.read.parquet('hdfs:/user/sy1880/df_val.parquet')
    val.createOrReplaceTempView('val')
#     test= spark.read.parquet('hdfs:/user/sy1880/df_test.parquet')
#     test.createOrReplaceTempView('test')
    print('Successfully loaded the data')


    import pyspark.sql.functions as func
    from pyspark.sql import Window
    from pyspark.sql import SparkSession

    from pyspark import StorageLevel
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.mllib.evaluation import RankingMetrics

    import numpy as np
    import pandas as pd
    import itertools
    from pyspark.ml.evaluation import RegressionEvaluator
    import time

    ranks = [200]  
    regParams = [1,10,20]   
    alphas = [80]
    maxIter = [10]
    
#     ranks = [i for i in range(20,80,20)]  
#     regParams = [10**i for i in range(-3,2)]   
#     alphas = [i for i in range(5,100,10)]
#     maxIter = [i for i in range(5,20,5)]

    params = [[a,b,c,d] for a,b,c,d in itertools.product(ranks, regParams, alphas, maxIter)]
    print('length of params',len(params))

    for sub in [train]:
        start_time = time.time()
        training = sub
        testing = val
        TOP = 500
        pk_high = 0
        
        unique_user_index_val = testing.select('user_index').distinct()
        # unique_user_index_test = test.select('user_index').distinct()

        res = []
        for rk,reg,alp,it in itertools.product(ranks, regParams, alphas, maxIter):
            als = ALS(userCol='user_index', itemCol='track_index', ratingCol='count', 
                      implicitPrefs=True, coldStartStrategy="drop", 
                      rank=rk, regParam=reg, alpha=alp, maxIter=it)
            model = als.fit(training)
            print('Successfully train the model')
            print("--- %s seconds ---" % (time.time() - start_time)) 
            
            userRecs = model.recommendForUserSubset(unique_user_index_val, TOP)
            print('Successfully get the recommendations')
            print("--- %s seconds ---" % (time.time() - start_time)) 
                        
            pred_tracks_rdd = userRecs.rdd.map(lambda row: (row['user_index'], 
                                                        [track_pred.track_index for track_pred in row['recommendations']]))

            print('Successfully transform the recommendations')
            # print(pred_tracks_rdd.take(1))

            w = Window.partitionBy('user_index').orderBy('count')
            true_tracks = testing.withColumn(
                'tracks', func.collect_list('track_index').over(w))\
                .groupBy('user_index')\
                .agg(func.max('tracks').alias('tracks'))
            true_tracks_rdd = true_tracks.rdd.map(tuple)

            print('Successfully transform the true values')
            # print(true_tracks_rdd.take(1))

            pred_and_true_tracks = pred_tracks_rdd.join(true_tracks_rdd)
            pred_and_true_tracks = pred_and_true_tracks.map(lambda tup: tup[1])
            print('Successfully put recommendations and true value together')
            # print(pred_and_true_tracks.take(1))

            metrics = RankingMetrics(pred_and_true_tracks)
            pk = metrics.precisionAt(TOP)
            res.append([rk,reg,alp,it,pk])
            print(pk)           

            print("Precision at 500 (validation) =", pk,"params (Rank, Reg, Alpha, Maxiter): ",[rk,reg,alp,it])
            if pk > pk_high:
                pk_high = pk
                param = [rk,reg,alp,it]
        
        print('best precision: ',pk_high,'best params (Rank, Reg, Alpha, Maxiter): ',param )
        print(res)



if __name__ == "__main__":
    # Create the spark session object
    MAX_MEMORY = "100g"

    spark = SparkSession \
    .builder \
    .appName("Foo") \
    .config("spark.executor.memory", MAX_MEMORY) \
    .config("spark.driver.memory", MAX_MEMORY) \
    .getOrCreate()
    # Call our main routine
    main(spark)