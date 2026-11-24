"""Arquitectura FT-Transformer (Gorishniy et al., 2021) implementada con Keras.

Idea: cada columna de la tabla se convierte en un vector (token). Un token especial [CLS]
se agrega al inicio, los bloques Transformer relacionan los tokens con atención, y la salida
del [CLS] se usa para clasificar la conexión en 5 clases.
"""
import keras
from keras import layers, ops


class TokenizadorNumerico(layers.Layer):
    """Convierte cada variable numérica x_i en un token: x_i * W_i + b_i."""

    def __init__(self, n_num, d, **kw):
        super().__init__(**kw)
        self.n_num, self.d = n_num, d

    def build(self, _):
        self.w = self.add_weight(shape=(self.n_num, self.d), initializer="glorot_uniform", name="w")
        self.b = self.add_weight(shape=(self.n_num, self.d), initializer="zeros", name="b")

    def call(self, x):                       # x: (lote, n_num)
        return ops.expand_dims(x, -1) * self.w + self.b   # (lote, n_num, d)


class TokenCLS(layers.Layer):
    """Agrega un token [CLS] aprendible al inicio de la secuencia."""

    def build(self, shape):
        self.cls = self.add_weight(shape=(1, 1, shape[-1]), initializer="glorot_uniform", name="cls")

    def call(self, x):
        cls = ops.tile(self.cls, [ops.shape(x)[0], 1, 1])
        return ops.concatenate([cls, x], axis=1)


def bloque_transformer(x, d, cabezas, dropout, nombre):
    """Bloque Transformer con pre-normalización: atención + red feed-forward, ambos con atajo."""
    h = layers.LayerNormalization(name=f"{nombre}_ln1")(x)
    h = layers.MultiHeadAttention(num_heads=cabezas, key_dim=d // cabezas, dropout=dropout,
                                  name=f"{nombre}_atencion")(h, h)
    x = layers.Add(name=f"{nombre}_suma1")([x, layers.Dropout(dropout)(h)])
    h = layers.LayerNormalization(name=f"{nombre}_ln2")(x)
    h = layers.Dense(d * 2, activation="relu", name=f"{nombre}_ffn1")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(d, name=f"{nombre}_ffn2")(h)
    return layers.Add(name=f"{nombre}_suma2")([x, layers.Dropout(dropout)(h)])


def crear_ft_transformer(n_num, cardinalidades, d=64, bloques=3, cabezas=8, dropout=0.1,
                         n_clases=5, lr=3e-4):
    ent_num = keras.Input(shape=(n_num,), name="numericas")
    ent_cat = keras.Input(shape=(len(cardinalidades),), dtype="int32", name="categoricas")

    tok_num = TokenizadorNumerico(n_num, d, name="tokens_numericos")(ent_num)
    tok_cat = [layers.Embedding(card, d, name=f"emb_cat_{i}")(ent_cat[:, i])
               for i, card in enumerate(cardinalidades)]
    tok_cat = layers.Concatenate(axis=1)([ops.expand_dims(t, 1) for t in tok_cat])
    x = layers.Concatenate(axis=1, name="tokens")([tok_num, tok_cat])   # (lote, 40, d)
    x = TokenCLS(name="token_cls")(x)                                    # (lote, 41, d)

    for i in range(bloques):
        x = bloque_transformer(x, d, cabezas, dropout, f"bloque{i + 1}")

    cls = layers.LayerNormalization(name="ln_final")(x[:, 0, :])
    cls = layers.Activation("relu")(cls)
    salida = layers.Dense(n_clases, activation="softmax", name="probabilidades")(cls)

    modelo = keras.Model([ent_num, ent_cat], salida, name="FT_Transformer")
    modelo.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo
