from contextlib import redirect_stdout
import io
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.callbacks import Callback

iris_text = "Scikit-learnのアイリスデータセットは、機械学習の分類問題においてよく使われるデータセットの一つです。\n このデータセットには、3つの種類のアヤメ（Setosa、Versicolour、およびVirginica）のそれぞれ50個のサンプルが含まれています。\n 各サンプルには、4つの特徴量（萼片の長さ、萼片の幅、花びらの長さ、花びらの幅）があります。アイリスデータセットは、1936年に英国の統計学者・植物学者であるロナルド・フィッシャーによって最初に収集され、その後、機械学習のアルゴリズムをテストするための基準データセットとして広く使用されるようになりました。"


class StreamlitCallback(Callback):
    def __init__(self, progress_bar, text_widget):
        super().__init__()
        self.progress_bar = progress_bar
        self.text_widget = text_widget
        self.epoch_scores = []

    def on_epoch_end(self, epoch, logs=None):
        value = (epoch + 1) / self.params['epochs']
        self.progress_bar.progress(value)

        # スコアを記録し、テキストウィジェットを更新
        score = f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}"
        self.epoch_scores.append(score)
        self.text_widget.text('\n'.join(self.epoch_scores))


def main():
    st.title("ニューラルネットワークの学習過程のリアルタイム可視化")

    # データセットの読み込み
    data = load_iris()
    X = data.data
    y = data.target
    iris_description = data['DESCR']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    st.markdown('### データの説明')
    st.write(iris_description)

    st.markdown('### 簡単な要約')
    st.write(iris_text)

    st.markdown('### データの確認')
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target_names[data.target]
    st.dataframe(df)

    # データの正規化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ハイパーパラメータ
    st.sidebar.title('ハイパーパラメータの設定')
    optimizer_select = st.sidebar.selectbox(
        "オプティマイザー(最適化アルゴリズム)選択", ("SGD", "RMSprop", "Adagrad", "Adam"))

    if optimizer_select == "SGD":
        optimizer = SGD()
    elif optimizer_select == "RMSprop":
        optimizer = RMSprop()
    elif optimizer_select == "Adagrad":
        optimizer = Adagrad()
    else:
        optimizer = Adam()

    epochs = st.sidebar.number_input("エポック数", 10, 500, step=10, value=100)
    batch_size = st.sidebar.number_input("バッチサイズ", 8, 256, step=8, value=64)
    activation_select = st.sidebar.selectbox(
        "活性化関数の選択", ("シグモイド関数", "ハイパボリックタンジェント関数", "reLu関数"))

    if activation_select == "SGD":
        activation = "sigmoid"
    elif activation_select == "RMSprop":
        activation = "tanh"
    else:
        activation = "relu"

    # ニューラルネットワークモデルの作成
    model = Sequential()
    model.add(Dense(64, activation=activation,
              input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation=activation))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # モデルの概要の表示
    if st.button("モデルの構造を表示"):
        # モデルの概要を取得するために標準出力を一時的にキャプチャ
        buf = io.StringIO()
        with redirect_stdout(buf):
            model.summary()

        # Streamlitにモデルの概要を表示
        st.subheader("モデルの概要")
        st.text(buf.getvalue())

    # 進捗バーとテキストウィジェットの設定
    progress_bar = st.sidebar.progress(0)
    epoch_scores_widget = st.empty()

    # カスタムコールバックの作成
    streamlit_callback = StreamlitCallback(progress_bar, epoch_scores_widget)

    # ニューラルネットワークモデルの学習
    if st.button("学習開始"):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, callbacks=[streamlit_callback])

        # 学習結果の可視化
        st.subheader("学習曲線")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='訓練データ')
        ax1.plot(history.history['val_loss'], label='検証データ')
        ax1.set_title('損失関数')
        ax1.legend()

        ax2.plot(history.history['accuracy'], label='訓練データ')
        ax2.plot(history.history['val_accuracy'], label='検証データ')
        ax2.set_title('学習データ・検証データの精度')
        ax2.legend()

        st.pyplot(fig)

        # 予測
        y_pred = np.argmax(model.predict(X_test), axis=-1)

        # 精度の計算
        acc = accuracy_score(y_test, y_pred)
        st.write(f"テストデータの精度: {acc:.4f}")

        # 混同行列の計算
        cm = confusion_matrix(y_test, y_pred)

        # 混同行列の表示
        st.subheader("混同行列")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)
        ax.set_xlabel('予測ラベル')
        ax.set_ylabel('真のラベル')

        st.pyplot(fig)


if __name__ == "__main__":
    main()
