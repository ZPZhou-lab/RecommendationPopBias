import tensorflow as tf
from core.dataset import PDADataset
from core.model import PDAMatrixFactorization
from evaluator.popularity import (
    calculate_recommendation_rate,
    calculate_popularity_vector
)
from evaluator.metrics import RecallCallback, PrecisionCallback, HitRateCallback, NDCGCallback
from utils import utils
# set GPU memory limit
utils.set_gpu_memory_limitation()

if __name__ == "__main__":
    dataset = PDADataset(
        train_path='~/datasets/Douban/movie_filter/train_cross_table.csv',
        valid_path='~/datasets/Douban/movie_filter/test_valid_cross_table.csv',
        test_path='~/datasets/Douban/movie_filter/test_test_cross_table.csv',
        batch_size=1024,
        neg_sample=1
    )

    # BPRMF
    model = PDAMatrixFactorization(
        num_users=dataset.NUM_USERS,
        num_items=dataset.NUM_ITEMS,
        embed_size=64,
        loss_func='BPR',
        add_bias=True,
        gamma=0.1,
        adjust=False, # PD or PDA
        l2_reg=1e-4
    )

    # learning rate scheduler
    # lr = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=1e-4,
    #     decay_steps=5000,
    #     alpha=0.1
    # )
    lr = 1e-4
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr))
    history = model.fit(
        dataset.create_tf_dataset(max_workers=4, max_steps=None),
        epochs=10,
        callbacks=[
            RecallCallback(dataset, top_k=20),
            PrecisionCallback(dataset, top_k=20),
            HitRateCallback(dataset, top_k=20),
            NDCGCallback(dataset, top_k=20)
        ]
    )

    # calculate recommendation rate
    popularity = calculate_popularity_vector(
        data=dataset.train.data, num_items=dataset.NUM_ITEMS
    )
    rec_items = model.recommend(dataset.train.users, top_k=50)
    rate = calculate_recommendation_rate(popularity, rec_items)
    print(f"====== Recommendation Rate ======\n{rate}")

    # save
    model.save_weights('./tests/douban/results/pdamf.h5')
    rate.to_csv('./tests/douban/results/pdamf_rr.csv')