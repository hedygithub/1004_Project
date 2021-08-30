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
    sub25= spark.read.parquet('hdfs:/user/sy1880/subsample25.parquet')
    sub25.createOrReplaceTempView('sub25')
#     print(sub1.head(),sub5.head(),sub25.head())

#     train= spark.read.parquet('hdfs:/user/sy1880/df_train.parquet')
#     train.createOrReplaceTempView('train')
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
    
    ranks = [i for i in range(20,200,20)]  
    regParams = [10**i for i in range(-3,1)]   
    alphas = [i for i in range(5,40,5)]
    maxIter = [i for i in range(5,30,5)]
#     ranks = [250]
#     regParams = [1]
#     alphas = [7]
#     maxIter = [20]

    params = [[a,b,c,d] for a,b,c,d in itertools.product(ranks, regParams, alphas, maxIter)]
    print('length of params',len(params))

    for sub in [sub25]:
        training = sub
        testing = val
        TOP = 500
        pk_high = 0

        mds = []
        alss = []
        for rk,reg,alp,it in itertools.product(ranks, regParams, alphas, maxIter):
            als = ALS(userCol="user_index", itemCol="track_index", ratingCol="count",
                    coldStartStrategy="drop",
                    implicitPrefs=True, rank=rk, regParam=reg, alpha=alp, maxIter=it)
            alss.append(als)
            model = als.fit(training)
            mds.append(model)
        print(len(mds),len(alss))

        users = testing.select(alss[0].getUserCol()).distinct()
        w = Window.partitionBy('user_index').orderBy('count')
#         print('num_testing_users',users.count())

        for i in range(len(mds)):
            userSubsetRecs = mds[i].recommendForUserSubset(users, TOP)
            # userSubsetRecs.persist(StorageLevel.MEMORY_ONLY)
            userRecs = userSubsetRecs
            
            pred_tracks = []
            for user, tracks in userRecs.collect():
                predict_tracks = [i[0] for i in tracks]
                pred_tracks.append((user, predict_tracks))
            pred_tracks_rdd = spark.sparkContext.parallelize(pred_tracks)

            true_tracks = testing.withColumn(
                'tracks', func.collect_list('track_index').over(w))\
                .groupBy('user_index')\
                .agg(func.max('tracks').alias('tracks'))
            true_tracks_rdd = true_tracks.rdd.map(tuple)

            pred_and_true_tracks = pred_tracks_rdd.join(true_tracks_rdd)
            pred_and_true_tracks = pred_and_true_tracks.map(lambda tup: tup[1])

            metrics = RankingMetrics(pred_and_true_tracks)
            pk = metrics.precisionAt(500)
            print("Precision at 5 (validation) =", pk,"params (Rank, Reg, Alpha, Maxiter): ",params[i])
            if pk > pk_high:
                pk_high = pk
                bestModel = mds[i]
                param = params[i]
        print('best precision: ',pk_high,'best params (Rank, Reg, Alpha, Maxiter): ',param )

            # ndcg = metrics.ndcgAt(10)
            # print("NDCG at 10 (validation) =", ndcg,"params (Rank, Reg, Alpha, Maxiter): ",params[i])
            # if ndcg > ndcg_high:
            #     ndcg_high = ndcg
            #     bestModel = mds[i]
            #     param = params[i]
        # print('best precision: ',ndcg_high,'best params (Rank, Reg, Alpha, Maxiter): ',param )

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