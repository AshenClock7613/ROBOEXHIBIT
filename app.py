import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rdflib
from rdflib.plugins.sparql.parser import parseQuery
from huggingface_hub import InferenceClient
import re
# ---------------------------------------------------------------------------
# CONFIGURAZIONE LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,  # Utilizziamo il livello DEBUG per un log più dettagliato
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COSTANTI / CHIAVI / MODELLI
# ---------------------------------------------------------------------------
# Nota: HF_API_KEY deve essere impostata a una chiave valida di Hugging Face.
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    # Se la chiave API non è impostata, solleva un errore
    logger.error("HF_API_KEY non impostata.")
    raise EnvironmentError("HF_API_KEY non impostata.")

# Nome del modello Hugging Face per generare query SPARQL e risposte finali
HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Nome del modello Hugging Face per rilevamento lingua
LANG_DETECT_MODEL = "papluca/xlm-roberta-base-language-detection"

# Prefisso per i modelli di traduzione su Hugging Face
TRANSLATOR_MODEL_PREFIX = "Helsinki-NLP/opus-mt"

# ---------------------------------------------------------------------------
# INIZIALIZZAZIONE CLIENT HUGGING FACE (una volta sola)
# ---------------------------------------------------------------------------
"""
Qui inizializziamo i client necessari. In questo modo, evitiamo di istanziare
continuamente nuovi oggetti InferenceClient a ogni chiamata delle funzioni.
- hf_generation_client: per generare query SPARQL e risposte stile "guida museale"
- lang_detect_client: per rilevare la lingua della domanda e della risposta
"""
try:
    logger.info("[Startup] Inizializzazione client HF per generazione (modello di LLM).")
    hf_generation_client = InferenceClient(
        token=HF_API_KEY,
        model=HF_MODEL
    )
    logger.info("[Startup] Inizializzazione client HF per rilevamento lingua.")
    lang_detect_client = InferenceClient(
        token=HF_API_KEY,
        model=LANG_DETECT_MODEL
    )
except Exception as ex:
    logger.error(f"Errore inizializzazione dei client Hugging Face: {ex}")
    raise HTTPException(status_code=500, detail="Impossibile inizializzare i modelli Hugging Face.")

# ---------------------------------------------------------------------------
# CARICAMENTO ONTOLOGIA
# ---------------------------------------------------------------------------
"""
Carichiamo il file RDF/XML contenente l'ontologia del museo. Questo file è 
fondamentale per l'esecuzione di query SPARQL, in quanto definisce le classi,
le proprietà e le istanze presenti nell'ontologia del museo.
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RDF_FILE = os.path.join(BASE_DIR, "Ontologia_corretto-2.rdf")

ontology_graph = rdflib.Graph()
try:
    logger.info(f"Caricamento ontologia da file: {RDF_FILE}")
    # Indichiamo che l'ontologia è in formato RDF/XML
    ontology_graph.parse(RDF_FILE, format="xml")
    logger.info("Ontologia RDF caricata correttamente (formato XML).")
except Exception as e:
    logger.error(f"Errore nel caricamento dell'ontologia: {e}")
    raise e
# ---------------------------------------------------------------------------
# DEFINIZIONE DELL'APP FASTAPI
# ---------------------------------------------------------------------------
app = FastAPI()
# ---------------------------------------------------------------------------
# Pydantic Model per la richiesta
# ---------------------------------------------------------------------------
class AssistantRequest(BaseModel):
    """
    Questo modello Pydantic definisce lo schema della richiesta che 
    riceverà l'endpoint /assistant. Contiene:
    - message: la domanda del visitatore
    - max_tokens: max di token per le risposte (di default 512)
    - temperature: temperatura di generazione (di default 0.5)
    """
    message: str
    max_tokens: int = 512
    temperature: float = 0.5

# ---------------------------------------------------------------------------
# FUNZIONI DI SUPPORTO (Prompts, validazione SPARQL, correzioni, ecc.)
# ---------------------------------------------------------------------------
def create_system_prompt_for_classification(ontology_turtle:str) -> str:
    prompt = f"""
SEI UN CLASSIFICATORE DI DOMANDE NEL CONTESTO DI UN MUSEO. 
DEVI ANALIZZARE LA DOMANDA DELL'UTENTE E, BASANDOTI SUL CONTESTO DELL'ONTOLOGIA (CHE TI VIENE FORNITA QUI SOTTO), RISPONDERE SOLAMENTE CON "SI" SE LA DOMANDA È PERTINENTE AL MUSEO, O CON "NO" SE LA DOMANDA NON È PERTINENTE.
RICORDA:
- "PERTINENTE" SIGNIFICA CHE LA DOMANDA RIGUARDA OPERE D'ARTE, ESPOSTE, BIGLIETTI, VISITATORI, CURATORI, RESTAURI, STANZE E TUTTI GLI ASPETTI RELATIVI ALL'AMBIENTE MUSEALE.
- "NON PERTINENTE" SIGNIFICA CHE LA DOMANDA RIGUARDA ARGOMENTI ESTERNI AL CONTESTO MUSEALE (PER ESEMPIO, TECNOLOGIA, SPORT, POLITICA, CUCINA, ECC.).
NON FORNIRE ULTERIORI SPIEGAZIONI O COMMENTI: LA TUA RISPOSTA DEVE ESSERE ESCLUSIVAMENTE "SI" O "NO".

ONTOLOGIA (TURTLE/XML/ALTRO FORMATO):
{ontology_turtle}

FINE ONTOLOGIA.

"""
    return prompt
def create_system_prompt_for_sparql(ontology_turtle: str) -> str:
    """
    Genera il testo di prompt che istruisce il modello su come costruire
    SOLO UNA query SPARQL, in un'unica riga, o in alternativa 'NO_SPARQL' 
    se la domanda non è pertinente all'ontologia. Il prompt include regole 
    di formattazione, esempi di domanda-risposta SPARQL e regole rigorose 
    per la gestione della posizione dell'opera.
    
    Parametri:
    - ontology_turtle: una stringa con l'ontologia in formato Turtle (o simile).
    
    Ritorna:
    - Il testo da usare come "system prompt" per il modello generativo.
    """
    prompt = f"""SEI UN GENERATORE DI QUERY SPARQL PER L'ONTOLOGIA DI UN MUSEO.
DEVI GENERARE SOLO UNA QUERY SPARQL (IN UNA SOLA RIGA) SE LA DOMANDA RIGUARDA INFORMAZIONI NELL'ONTOLOGIA.
SE LA DOMANDA NON È ATTINENTE, RISPONDI 'NO_SPARQL'.
REGOLE SINTATTICHE RIGOROSE:
1) Usare: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#>
2) Query in UNA SOLA RIGA (niente a capo), forma: PREFIX progettoMuseo: <...> SELECT ?x WHERE {{ ... }} LIMIT N
3) Attento agli spazi:
   - Dopo SELECT: es. SELECT ?autore
   - Tra proprietà e variabile: es. progettoMuseo:autoreOpera ?autore .
   - Non incollare il '?' a 'progettoMuseo:'.
   - Ogni tripla termina con un punto.
4) Se non puoi generare una query valida, rispondi solo 'NO_SPARQL'.

Esempi di Domande Specifiche e relative Query:
1) Utente: Chi ha creato l'opera 'Afrodite di Milo'?
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?autore WHERE {{ progettoMuseo:AfroditeDiMilo progettoMuseo:autoreOpera ?autore . }} LIMIT 10
2) Utente: Quali sono le tecniche utilizzate nelle opere?
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera ?tecnica WHERE {{ ?opera progettoMuseo:tecnicaOpera ?tecnica . }} LIMIT 100
3) Utente: Quali sono le dimensioni delle opere?
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera ?dimensione WHERE {{ ?opera progettoMuseo:dimensioneOpera ?dimensione . }} LIMIT 100
4) Utente: Quali opere sono esposte nella stanza Greca?
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera WHERE {{ progettoMuseo:StanzaGrecia progettoMuseo:Espone ?opera . }} LIMIT 100
5) Utente: Quali sono le proprietà e i tipi delle proprietà nell'ontologia?
   Risposta: PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT DISTINCT ?property ?type WHERE {{ ?property rdf:type ?type . FILTER(?type IN (owl:ObjectProperty, owl:DatatypeProperty)) }}
6) Utente: Recupera tutti i biglietti e i tipi di biglietto.
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?biglietto ?tipoBiglietto WHERE {{ ?biglietto rdf:type progettoMuseo:Biglietto . ?biglietto progettoMuseo:tipoBiglietto ?tipoBiglietto . }} LIMIT 100
7) Utente: Recupera tutti i visitatori e i tour a cui partecipano.
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?visitatore ?tour WHERE {{ ?visitatore progettoMuseo:Partecipazione_a_Evento ?tour . }} LIMIT 100
8) Utente: Recupera tutte le stanze tematiche e le opere esposte.
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?stanza ?opera WHERE {{ ?stanza rdf:type progettoMuseo:Stanza_Tematica . ?stanza progettoMuseo:Espone ?opera . }} LIMIT 100
9) Utente: Recupera tutte le opere con materiale 'Marmo'.
   Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera WHERE {{ ?opera progettoMuseo:materialeOpera "Marmo"@it . }} LIMIT 100
10) Utente: Recupera tutti i visitatori con data di nascita dopo il 2000.
    Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?visitatore WHERE {{ ?visitatore rdf:type progettoMuseo:Visitatore_Individuale . ?visitatore progettoMuseo:dataDiNascitaVisitatore ?data . FILTER(?data > "2000-01-01T00:00:00"^^xsd:dateTime) . }} LIMIT 100

NUOVE REGOLE RIGUARDANTI LA POSIZIONE DELL'OPERA:
- SE la domanda include richieste come "fammi vedere l'opera X", "mi fai vedere l'opera X", "portami all'opera X", "voglio vedere l'opera X" o simili, la query SPARQL DEVE restituire la posizione dell'opera includendo la proprietà progettoMuseo:posizioneOpera.
- SE la domanda include espressioni come "dove si trova l'opera X", "ubicazione dell'opera X" o simili, la query SPARQL NON deve restituire la proprietà progettoMuseo:posizioneOpera, limitandosi al massimo a riportare la relazione esistente.

Esempi Aggiuntivi:
11) Utente: Fammi vedere l'opera 'Discobolo'.
    Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera ?posizione WHERE {{ progettoMuseo:Discobolo progettoMuseo:posizioneOpera ?posizione . }} LIMIT 10
12) Utente: Mi fai vedere l'opera 'Gioconda'?
    Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera ?posizione WHERE {{ progettoMuseo:Gioconda progettoMuseo:posizioneOpera ?posizione . }} LIMIT 10
13) Utente: Portami all'opera 'Autoritratto'.
    Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera ?posizione WHERE {{ progettoMuseo:Autoritratto progettoMuseo:posizioneOpera ?posizione . }} LIMIT 10
14) Utente: Dove si trova l'opera 'La Notte Stellata'?
    Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera WHERE {{ progettoMuseo:LaNotteStellata ?rel ?info . FILTER(?rel != progettoMuseo:posizioneOpera) }} LIMIT 10
15) Utente: Qual è l'ubicazione dell'opera 'Ramo Di Mandorlo Fiorito'?
    Risposta: PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> SELECT ?opera WHERE {{ progettoMuseo:RamoDiMandorloFiorito ?rel ?info . FILTER(?rel != progettoMuseo:posizioneOpera) }} LIMIT 10

ECCO L'ONTOLOGIA (TURTLE) PER CONTESTO:
{ontology_turtle}
FINE ONTOLOGIA.
"""
    logger.debug("[create_system_prompt_for_sparql] Prompt generato con esempi originali e nuove regole rigorose sulla posizione.")
    return prompt



def classify_and_translate(question_text: str, model_answer_text: str):
    """
    Classifica la lingua della domanda e della risposta, quindi traduce la risposta
    se la lingua è diversa da quella della domanda. L'idea è di restituire una 
    risposta nella stessa lingua dell'utente.
    
    Parametri:
    - question_text: Testo della domanda dell'utente.
    - model_answer_text: Risposta del modello (in qualsiasi lingua).
    
    Restituisce:
    - La risposta tradotta nella lingua della domanda o la risposta originale
      se entrambe le lingue coincidono.
    NB: Qui l'oggetto 'lang_detect_client' (per rilevamento lingua) è già 
    stato inizializzato all'avvio dell'app. Mentre il 'translator_client'
    viene creato 'al volo' poiché la direzione di traduzione dipende 
    dalle due lingue effettive.
    """
    # Rileva la lingua della domanda
    try:
        question_lang_result = lang_detect_client.text_classification(text=question_text)
        question_lang = question_lang_result[0]['label']
        logger.info(f"[LangDetect] Lingua della domanda: {question_lang}")
    except Exception as e:
        logger.error(f"Errore nel rilevamento della lingua della domanda: {e}")
        question_lang = "en"  # Fallback se non riusciamo a rilevare la lingua
    # Rileva la lingua della risposta
    try:
        answer_lang_result = lang_detect_client.text_classification(text=model_answer_text)
        answer_lang = answer_lang_result[0]['label']
        logger.info(f"[LangDetect] Lingua della risposta: {answer_lang}")
    except Exception as e:
        logger.error(f"Errore nel rilevamento della lingua della risposta: {e}")
        answer_lang = "it"  # Fallback se non riusciamo a rilevare la lingua

    # Se domanda e risposta sono nella stessa lingua, non traduciamo
    if question_lang == answer_lang:
        logger.info("[Translate] Nessuna traduzione necessaria: stessa lingua.")
        return model_answer_text

    # Altrimenti, costruiamo "al volo" il modello di traduzione appropriato 
    # (es: "Helsinki-NLP/opus-mt-en-it", "Helsinki-NLP/opus-mt-fr-en", ecc.)
    translator_model = f"{TRANSLATOR_MODEL_PREFIX}-{answer_lang}-{question_lang}"
    translator_client = InferenceClient(
        token=HF_API_KEY,
        model=translator_model
    )
    # Traduzione della risposta
    try:
        translation_result = translator_client.translation(text=model_answer_text)
        translated_answer = translation_result["translation_text"]
    except Exception as e:
        logger.error(f"Errore nella traduzione {answer_lang} -> {question_lang}: {e}")
        # Se fallisce, restituiamo la risposta originale come fallback
        translated_answer = model_answer_text

    return translated_answer


def create_system_prompt_for_guide() -> str:
    """
    Genera un testo di prompt che istruisce il modello a rispondere 
    come "guida museale virtuale", in modo breve (~50 parole), riassumendo 
    i risultati SPARQL (se presenti) o fornendo comunque una risposta 
    in base alle conoscenze pregresse.
    """
    prompt = (
        "SEI UNA GUIDA MUSEALE VIRTUALE. "
        "RISPONDI IN MODO DISCORSIVO E NATURALE (circa 100 parole), SENZA SALUTI O INTRODUZIONI PROLISSE. "
        "SE HAI RISULTATI SPARQL, USALI; SE NON CI SONO, RISPONDI BASANDOTI SULLE TUE CONOSCENZE. "
        "QUALORA LA DOMANDA CONTENGA ESPRESSIONI COME 'fammi vedere', 'portami', 'mi fai vedere', O SIMILI, "
        "TRADUCI LA RICHIESTA IN UN INVITO ALL'AZIONE, AD ESEMPIO 'Adesso ti accompagno all'opera', "
        "SENZA RIPETERE DETTAGLI SPAZIALI O TECNICI (ES. COORDINATE, DISTANZE, POSITIONI FISICHE). "
        "PER ALTRE DOMANDE, RISPONDI IN MODO DESCRITTIVO MA SENZA INCLUDERE INFORMAZIONI TECNICHE SULLA POSIZIONE."
    )
    logger.debug("[create_system_prompt_for_guide] Prompt per la risposta guida museale generato.")
    return prompt


def correct_sparql_syntax_advanced(query: str) -> str:
    """
    Applica correzioni sintattiche (euristiche) su una query SPARQL eventualmente
    mal formattata, generata dal modello. 
    Passi:
      1. Rimuove newline.
      2. Verifica l'esistenza di 'PREFIX progettoMuseo:' e lo aggiunge se mancante.
      3. Inserisce spazi dopo SELECT, WHERE (se mancanti).
      4. Se c'è 'progettoMuseo:autoreOpera?autore' lo trasforma in 'progettoMuseo:autoreOpera ?autore'.
      5. Rimuove spazi multipli.
      6. Aggiunge '.' prima di '}' se manca.
      7. Aggiunge la clausola WHERE se non presente.
    
    Parametri:
    - query: stringa con la query SPARQL potenzialmente mal formattata.
    Ritorna:
    - La query SPARQL corretta se possibile, in singola riga.
    """
    original_query = query
    logger.debug(f"[correct_sparql_syntax_advanced] Query originaria:\n{original_query}")

    # 1) Rimuoviamo newline e normalizziamo a una singola riga
    query = query.replace('\n', ' ').replace('\r', ' ')

    # 2) Se manca il PREFIX museo, lo aggiungiamo in testa
    if 'PREFIX progettoMuseo:' not in query:
        logger.debug("[correct_sparql_syntax_advanced] Aggiungo PREFIX progettoMuseo.")
        query = (
            "PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> "
            + query
        )

    # 3) Spazio dopo SELECT se manca (SELECT?autore => SELECT ?autore)
    query = re.sub(r'(SELECT)(\?|\*)', r'\1 \2', query, flags=re.IGNORECASE)

    # 4) Spazio dopo WHERE se manca (WHERE{ => WHERE {)
    query = re.sub(r'(WHERE)\{', r'\1 {', query, flags=re.IGNORECASE)

    # 5) Correggiamo le incollature: progettoMuseo:autoreOpera?autore => progettoMuseo:autoreOpera ?autore
    query = re.sub(r'(progettoMuseo:\w+)\?(\w+)', r'\1 ?\2', query)

    # 6) Rimuoviamo spazi multipli
    query = re.sub(r'\s+', ' ', query).strip()

    # 7) Aggiungiamo '.' prima di '}' se manca
    query = re.sub(r'(\?\w+)\s*\}', r'\1 . }', query)

    # 8) Se manca la clausola WHERE, la aggiungiamo
    if 'WHERE' not in query.upper():
        query = re.sub(r'(SELECT\s+[^\{]+)\{', r'\1 WHERE {', query, flags=re.IGNORECASE)

    # 9) Pulizia spazi superflui
    query = re.sub(r'\s+', ' ', query).strip()

    logger.debug(f"[correct_sparql_syntax_advanced] Query dopo correzioni:\n{query}")
    return query


def is_sparql_query_valid(query: str) -> bool:
    """
    Verifica la validità sintattica di una query SPARQL usando rdflib.
    Ritorna True se la query è sintatticamente corretta, False altrimenti.
    """
    logger.debug(f"[is_sparql_query_valid] Validazione SPARQL: {query}")
    try:
        parseQuery(query)
        logger.debug("[is_sparql_query_valid] Query SPARQL sintatticamente corretta.")
        return True
    except Exception as ex:
        logger.warning(f"[is_sparql_query_valid] Query non valida: {ex}")
        return False

# ---------------------------------------------------------------------------
# ENDPOINT UNICO: /assistant
# ---------------------------------------------------------------------------
@app.post("/assistant")
def assistant_endpoint(req: AssistantRequest):
    """
    Endpoint che gestisce l'intera pipeline:
     1) Genera una query SPARQL dal messaggio dell'utente (prompt dedicato).
     2) Verifica la validità della query e, se valida, la esegue sull'ontologia RDF.
     3) Crea un "prompt da guida museale" e genera una risposta finale breve (max ~50 parole).
     4) Eventualmente, traduce la risposta nella lingua dell'utente.
    
    Parametri:
    - req (AssistantRequest): un oggetto contenente:
       - message (str): la domanda dell'utente
       - max_tokens (int, opzionale): numero massimo di token per la generazione
       - temperature (float, opzionale): temperatura per la generazione
    
    Ritorna:
    - Un JSON con:
       {
         "query": <la query SPARQL generata o None>,
         "response": <la risposta finale in linguaggio naturale>
       }
    """
    logger.info("Ricevuta chiamata POST su /assistant")
    # Estraggo i campi dal body della richiesta
    user_message = req.message
    max_tokens = req.max_tokens
    temperature = req.temperature
    logger.debug(f"Parametri utente: message='{user_message}', max_tokens={max_tokens}, temperature={temperature}")
    # -----------------------------------------------------------------------
    # STEP 1: Generazione della query SPARQL
    # -----------------------------------------------------------------------
    try:
        # Serializziamo l'ontologia in XML per fornirla al prompt (anche se si chiama 'turtle' va bene così).
        ontology_turtle = ontology_graph.serialize(format="xml")
        logger.debug("Ontologia serializzata con successo (XML).")
    except Exception as e:
        logger.warning(f"Impossibile serializzare l'ontologia in formato XML: {e}")
        ontology_turtle = ""

    # Creiamo il prompt di sistema per la generazione SPARQL
    system_prompt_sparql = create_system_prompt_for_sparql(ontology_turtle)
    # Chiamata al modello per generare la query SPARQL
    try:
        logger.debug("[assistant_endpoint] Chiamata HF per generare la query SPARQL...")
        gen_sparql_output = hf_generation_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt_sparql},
                {"role": "user", "content": user_message}
            ],
            max_tokens=512,       # max_tokens per la generazione della query
            temperature=0.2       # temperatura bassa per avere risposte più "deterministiche"
        )
        possible_query = gen_sparql_output["choices"][0]["message"]["content"].strip()
        logger.info(f"[assistant_endpoint] Query generata dal modello: {possible_query}")
    except Exception as ex:
        logger.error(f"Errore nella generazione della query SPARQL: {ex}")
        # Se fallisce la generazione, consideriamo la query come "NO_SPARQL"
        possible_query = "NO_SPARQL"

    # Verifichiamo se la query è "NO_SPARQL"
    if possible_query.upper().startswith("NO_SPARQL"):
        generated_query = None
        logger.debug("[assistant_endpoint] Modello indica 'NO_SPARQL', quindi nessuna query generata.")
    else:
        # Applichiamo la correzione avanzata
        advanced_corrected = correct_sparql_syntax_advanced(possible_query)
        # Verifichiamo la validità della query
        if is_sparql_query_valid(advanced_corrected):
            generated_query = advanced_corrected
            logger.debug(f"[assistant_endpoint] Query SPARQL valida dopo correzione avanzata: {generated_query}")
        else:
            logger.debug("[assistant_endpoint] Query SPARQL non valida. Verrà ignorata.")
            generated_query = None

    # -----------------------------------------------------------------------
    # STEP 2: Esecuzione della query, se disponibile
    # -----------------------------------------------------------------------
    results = []
    point = ""  # variabile per contenere il punto (inizialmente nessuno)
    if generated_query:
        logger.debug(f"[assistant_endpoint] Esecuzione della query SPARQL:\n{generated_query}")
        try:
            query_result = ontology_graph.query(generated_query)
            results = list(query_result)
            logger.info(f"[assistant_endpoint] Query eseguita con successo. Numero risultati = {len(results)}")
        except Exception as ex:
            logger.error(f"[assistant_endpoint] Errore nell'esecuzione della query: {ex}")
            results = []
    # -----------------------------------------------------------------------
    # STEP 3: Generazione della risposta finale stile "guida museale"
    # -----------------------------------------------------------------------
    system_prompt_guide = create_system_prompt_for_guide()
    if generated_query and results:
        # Caso: query generata + risultati SPARQL
        # Convertiamo i risultati in una stringa più leggibile
        results_str = "\n".join(
            f"{idx+1}) " + ", ".join(f"{var}={row[var]}" for var in row.labels)
            for idx, row in enumerate(results)
        )
        # Estraiamo il primo punto valido (se presente)
        for row in results:
        # Verifica se la riga contiene la variabile "p" e ha un valore non vuoto
            if "p" in row.labels and row["p"] is not None and str(row["p"]).strip() != "":
                point = str(row["p"]).strip()
                break  # Esci dal ciclo non appena trovi il primo punto valido
            # Se "p" non è presente, controlla la variabile "posizione"
            elif "posizione" in row.labels and row["posizione"] is not None and str(row["posizione"]).strip() != "":
                point = str(row["posizione"]).strip()
                break  # Esci dal ciclo se trovi il primo punto valido
        second_prompt = (
            f"{system_prompt_guide}\n\n"
            f"Domanda utente: {user_message}\n"
            f"Query generata: {generated_query}\n"
            f"Risultati:\n{results_str}\n"
            "Rispondi in modo breve (max ~50 parole)."
        )
        logger.debug("[assistant_endpoint] Prompt di risposta con risultati SPARQL.")
    elif generated_query and not results:
        # Caso: query valida ma 0 risultati
        second_prompt = (
            f"{system_prompt_guide}\n\n"
            f"Domanda utente: {user_message}\n"
            f"Query generata: {generated_query}\n"
            "Nessun risultato dalla query. Prova comunque a rispondere con le tue conoscenze."
        )
        logger.debug("[assistant_endpoint] Prompt di risposta: query valida ma senza risultati.")
    else:
        # Caso: nessuna query generata
        second_prompt = (
            f"{system_prompt_guide}\n\n"
            f"Domanda utente: {user_message}\n"
            "Nessuna query SPARQL generata. Rispondi come puoi, riarrangiando le tue conoscenze."
        )
        logger.debug("[assistant_endpoint] Prompt di risposta: nessuna query generata.")
    # Chiamata finale al modello per la risposta "guida museale"
    try:
        logger.debug("[assistant_endpoint] Chiamata HF per generare la risposta finale...")
        final_output = hf_generation_client.chat.completions.create(
            messages=[
                {"role": "system", "content": second_prompt},
                {"role": "user", "content": "Fornisci la risposta finale."}
            ],
            max_tokens=512,
            temperature=0.5
        )
        final_answer = final_output["choices"][0]["message"]["content"].strip()
        logger.info(f"[assistant_endpoint] Risposta finale generata: {final_answer}")
    except Exception as ex:
        logger.error(f"Errore nella generazione della risposta finale: {ex}")
        raise HTTPException(status_code=500, detail="Errore nella generazione della risposta in linguaggio naturale.")

    # -----------------------------------------------------------------------
    # STEP 4: Traduzione (se necessario)
    # -----------------------------------------------------------------------
    final_ans = classify_and_translate(user_message, final_answer)
    final_ans = final_ans.replace('\\"', "").replace('\"', "")
    # -----------------------------------------------------------------------
    # Restituzione in formato JSON
    # -----------------------------------------------------------------------
    logger.debug("[assistant_endpoint] Fine elaborazione, restituzione risposta JSON.")
    return {
        "query": generated_query,
        "response": final_ans,
        "point": point
    }
# ---------------------------------------------------------------------------
# ENDPOINT DI TEST / HOME
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    """
    Endpoint di test per verificare se l'applicazione è in esecuzione.
    """
    logger.debug("Chiamata GET su '/' - home.")
    return {
        "message": "Endpoint attivo. Esempio di backend per generare query SPARQL e risposte guida museale."
    }
# ---------------------------------------------------------------------------
# ENDPOINT QUERY STANZE /QUERY_STANZE
# ---------------------------------------------------------------------------
@app.get("/query_stanze")
def query_stanze_endpoint():
    """
    Endpoint per restituire le stanze con le opere esposte e relativi punti.
    
    La query utilizza la seguente struttura di output:
    
    {
        "stanze": [
            {
                "roomName": "stanzaGrecia",
                "entries": [
                    { "nome": "AfroditeDiMilo", "punto": "(-1.171, 0, -0.004)" },
                    { "nome": "Discobolo", "punto": "(0, 0, 0.77)" },
                    ... 
                ]
            },
            {
                "roomName": "stanzaItalia",
                "entries": [
                    { "nome": "AmoreEPsiche", "punto": "(0.677, 0, 0)" },
                    { "nome": "David", "punto": "(0.586, 0, -0.1)" },
                    ... 
                ]
            },
            ... 
        ]
    }
    """
    # Definiamo la query in una singola riga
    query_str = (
        "PREFIX progettoMuseo: <http://www.semanticweb.org/lucreziamosca/ontologies/progettoMuseo#> "
        "SELECT ?stanza ?opera ?p WHERE { ?opera progettoMuseo:èEsposto ?stanza. "
        "OPTIONAL { ?opera progettoMuseo:posizioneOpera ?p. } }"
    )
    
    # Applichiamo eventuali correzioni sintattiche
    corrected_query = correct_sparql_syntax_advanced(query_str)
    logger.debug(f"[query_stanze_endpoint] Query corretta:\n{corrected_query}")
    
    # Verifica che la query sia sintatticamente corretta
    if not is_sparql_query_valid(corrected_query):
        logger.error("[query_stanze_endpoint] Query SPARQL non valida.")
        raise HTTPException(status_code=400, detail="Query SPARQL non valida.")
    
    try:
        query_result = ontology_graph.query(corrected_query)
        
        # Costruiamo un dizionario temporaneo per raggruppare le entry per stanza
        temp_dict = {}
        for row in query_result:
            # Estrazione del localName della stanza
            stanza_uri = str(row["stanza"])
            stanza_local = stanza_uri.split("#")[-1].strip() if "#" in stanza_uri else stanza_uri
            
            # Estrazione del localName dell'opera
            opera_uri = str(row["opera"])
            opera_local = opera_uri.split("#")[-1].strip() if "#" in opera_uri else opera_uri
            
            # Estrazione del punto (se presente)
            punto = str(row["p"]) if row["p"] is not None else ""
            
            # Crea il dizionario per l'opera
            opera_dict = {"nome": opera_local, "punto": punto}
            
            # Raggruppa le entry per stanza
            if stanza_local not in temp_dict:
                temp_dict[stanza_local] = []
            temp_dict[stanza_local].append(opera_dict)
        
        # Trasforma il dizionario in un array che segue la struttura richiesta:
        # ogni elemento dell'array è un oggetto con i campi "roomName" e "entries"
        stanze_list = []
        for room_name, entries in temp_dict.items():
            stanze_list.append({"roomName": room_name, "entries": entries})
        
        logger.info(f"[query_stanze_endpoint] Trovate {len(stanze_list)} stanze.")
    except Exception as ex:
        logger.error(f"[query_stanze_endpoint] Errore nell'esecuzione della query: {ex}")
        raise HTTPException(status_code=500, detail="Errore nell'esecuzione della query SPARQL.")
    
    # Restituisce il JSON seguendo la struttura attesa dal modello C#
    return {
        "stanze": stanze_list
    }

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Avvio dell'applicazione FastAPI sulla porta 8000, 
    utile se eseguito come script principale.
    """
    logger.info("Avvio dell'applicazione FastAPI.")
