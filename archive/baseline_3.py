# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
import getpass


def main(spark):
    '''Main routine for Project
    Parameters
    ----------
    spark : SparkSession object
    '''
    netid = getpass.getuser()
    print('Begin loading data')

    # Load the Data
    train = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    train.createOrReplaceTempView('train')
    val = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
    val.createOrReplaceTempView('val')
    test = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_test.parquet')
    test.createOrReplaceTempView('test')
    print('Successfully loaded the data')

    # Get Unique user and track
    FRACTION = 0.01
    unique_user, unique_user_val = (train.select('user_id')).distinct(), val.select('user_id').distinct()
    unique_track = train.select('track_id').union(val.select('track_id')).union(test.select('track_id')).distinct()

    # Subsample Based on userid
    subset_unique_user = unique_user.sample(withReplacement=False, fraction=FRACTION).union(unique_user_val).distinct()
    train.join(subset_unique_user, train.user_id == subset_unique_user.user_id, how='leftsemi')
    print('Successfully subsample the data')

    # Encode string to index============================================
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_index")
    tran_user = indexer_user.fit(unique_user)
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_index")
    tran_track = indexer_track.fit(unique_track)

    train = tran_user.transform(train)
    train = tran_track.transform(train)
    val = tran_user.transform(val)
    # test = tran_user.transform(test)
    val = tran_track.transform(val)
    # test = tran_track.transform(test)
    print('Successfully transformed the data')

    als = ALS(numItemBlocks=100, numUserBlocks=500, maxIter=4,
              userCol='user_index', itemCol='track_index', ratingCol='count', implicitPrefs=True,
              alpha=0.1, coldStartStrategy="drop")
    model = als.fit(train)
    print('Successfully fit the model')

    user_track_tuple = val.map(lambda x: (x.user_index, x.track_index))
    reals = val.map(lambda r: ((r.user, r.product), r.rating))
    predictions = als.predictAll(user_track_tuple).map(lambda r: ((r.user_index, r.track_index), r.count))
    scoreAndLabels = predictions.join(reals).map(lambda tup: tup[1])
    scoreAndLabels.show()

    # precision at k
    K = 10
    metrics = RegressionMetrics(scoreAndLabels)
    precision = metrics.precisionAt(K)
    print('Successfully evaluate the model')
    print('precision at {}, is {:.4f}'.format(K, precision))

    model.save(spark, 'hdfs:/user/{}/subsample252.parquet'.format(netid))
    print('Successfully save the model')




if __name__ == "__main__":
    # Create the spark session object
    MAX_MEMORY = "20g"

    spark = SparkSession \
        .builder \
        .appName("Foo") \
        .config("spark.executor.memory", MAX_MEMORY) \
        .config("spark.driver.memory", MAX_MEMORY) \
        .getOrCreate()
    # Call our main routine
    main(spark)
