import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega os dados
def carregar_dados():
    basics = pd.read_csv("title.basics.tsv.gz", sep="\t", dtype=str, compression="gzip", na_values="\\N")
    ratings = pd.read_csv("title.ratings.tsv.gz", sep="\t", dtype=str, compression="gzip", na_values="\\N")

    # Filtrar apenas filmes
    filmes = basics[basics["titleType"] == "movie"]
    filmes = filmes[["tconst", "primaryTitle", "genres"]].dropna()

    # Unir com avaliações
    filmes = filmes.merge(ratings, on="tconst")
    filmes["averageRating"] = filmes["averageRating"].astype(float)

    return filmes

def criar_vetor_generos(filmes):
    tfidf = TfidfVectorizer(token_pattern=r"[^,]+")
    tfidf_matrix = tfidf.fit_transform(filmes["genres"])
    return tfidf, tfidf_matrix

def recomendar(filmes, tfidf, tfidf_matrix, titulo, n=5):
    filmes_lower = filmes["primaryTitle"].str.lower()
    idx_lista = filmes[filmes_lower == titulo.lower()].index

    if len(idx_lista) == 0:
        possiveis = difflib.get_close_matches(titulo.lower(), filmes_lower, n=5, cutoff=0.6)
        return None, possiveis

    idx = idx_lista[0]
    query_vec = tfidf.transform([filmes.iloc[idx]["genres"]])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:n+1]
    return filmes.iloc[similar_indices][["primaryTitle", "genres", "averageRating"]], []

# Exemplo de uso
if __name__ == "__main__":
    print("Carregando dados...")
    filmes = carregar_dados()
    tfidf, tfidf_matrix = criar_vetor_generos(filmes)

    todos_generos = sorted(set(g for sublist in filmes["genres"].str.split(",") for g in sublist))

    while True:
        print("\nDigite 'sair' para encerrar.")
        titulo = input("\nDigite o nome de um filme: ").strip()
        if titulo.lower() == "sair":
            print("Encerrando. Até logo!")
            break
        
        print("\n----------------------------------------------------------------------------------")
        print("\nGêneros disponíveis:")
        print(", ".join(todos_generos))
        print("\n----------------------------------------------------------------------------------")
        genero_filtro = input("\nDeseja filtrar por gênero (pressione Enter para pular)? ").strip()
        

        nota_minima = input("Nota mínima (ex: 7.0)? (pressione Enter para pular): ").strip()
        nota_minima = float(nota_minima) if nota_minima else 0.0

        resultado, sugestoes = recomendar(filmes, tfidf, tfidf_matrix, titulo)

        if resultado is None and sugestoes:
            print("\nFilme não encontrado. Talvez você quis dizer:")
            for i, s in enumerate(sugestoes):
                print(f"{i + 1}. {s}")
            try:
                escolha = int(input("Escolha uma opção (número) ou 0 para cancelar: "))
                if escolha < 1 or escolha > len(sugestoes):
                    print("Cancelado.")
                    continue
                resultado, _ = recomendar(filmes, tfidf, tfidf_matrix, sugestoes[escolha - 1])
            except ValueError:
                print("Entrada inválida. Cancelado.")
                continue

        if resultado is not None and not resultado.empty:
            if genero_filtro:
                resultado = resultado[resultado["genres"].str.contains(genero_filtro, case=False, na=False)]
            resultado = resultado[resultado["averageRating"] >= nota_minima]

            if resultado.empty:
                print("\nNenhum filme encontrado com os filtros aplicados.")
            else:
                print("\nFilmes recomendados:")
                for _, row in resultado.iterrows():
                    print(f"- {row['primaryTitle']} ({row['genres']}) - Nota: {row['averageRating']}")
        else:
            print("Nenhuma recomendação encontrada.")
