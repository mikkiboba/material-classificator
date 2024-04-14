import time
from training import execute_model


def main() -> None:
    execute_model(learning_rate=0.0001, batch_size=64, n_epochs=15)


if __name__ == "__main__":
    now = time.time()
    main()
    print(f'-- EXECUTION TIME: {time.time()-now} seconds')
