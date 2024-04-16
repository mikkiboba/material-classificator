import time
from training import execute_model_complete, execute_model_noisy_noisy


def main() -> None:
    #execute_model(learning_rate=0.0001, batch_size=24, n_epochs=20, load_dataset=False)
    #execute_model_noisy_noisy(learning_rate=0.0001, batch_size=16, n_epochs=20, load_dataset=True)
    execute_model_complete(learning_rate=0.001, batch_size=64, n_epochs=120, load_dataset=True)


if __name__ == "__main__":
    now = time.time()
    main()
    print(f'-- EXECUTION TIME: {time.time()-now} seconds')
