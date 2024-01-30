import pickle
import sqlite3

class Logger():
    def __init__(self, db_file="model.db", delete_table=True):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        if delete_table:
            self.cursor.execute("DROP TABLE IF EXISTS results;")
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS results (
                EPOCH INTEGER NOT NULL,
                REWARD INT NOT NULL,
                T_STEPS INT NOT NULL,
                PRIMARY KEY (EPOCH)
            );"""
        )

    def record_run(self, epoch, reward, t_steps, model):
        self.cursor.execute(
            """
            INSERT INTO results (EPOCH, REWARD, T_STEPS)
            VALUES 
            (?, ?, ?)
            """, (epoch, reward, t_steps)
        )
        if epoch % 1000 == 0:
            pickle.dump(model, "_".join([epoch, reward, t_steps]))

    def commit(self):
        self.conn.commit()
