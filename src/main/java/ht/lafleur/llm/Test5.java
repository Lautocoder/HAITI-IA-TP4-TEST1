package ht.lafleur.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.List;
import java.util.Scanner;

import static ht.lafleur.llm.Utils.getKey;

public class Test5 {

    public static void main(String[] args) {

        Utils.configureLogger();

//        String geminiKey = getKey("GEMINI_KEY");
        String claudeKey= getKey("CLAUDE_KEY");
        String tavilyKey= getKey("TAVILY_KEY");

        String documentPath = "rag.pdf";

//        ChatModel modelGemini = GoogleAiGeminiChatModel.builder()
//                .apiKey(geminiKey)
//                .modelName("gemini-3-flash")
//                .temperature(0.3)
//                .logRequests(true)
//                .logResponses(true)
//                .build();

        ChatModel modelClaude = AnthropicChatModel.builder()
                .apiKey(claudeKey)
                .modelName("claude-sonnet-4-6")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();


        // Création d'un Document à partir du fichier PDF avec  un parseur de fichier PDF fourni par LangChain4j
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(documentPath, parser);

        // Découpage du document en morceaux
        DocumentSplitter documentSplitter = DocumentSplitters.recursive(512,30);
        List<TextSegment> segments = documentSplitter.split(document);

        // Création d'un modèle d'embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Création des embeddings pour les segments
        var embeddings = embeddingModel.embedAll(segments).content();


        /*/ Utilisation d'une base vectorielle Qdrant pour stocker les embeddings et les segments associés.
        EmbeddingStore<TextSegment> embeddingStore = QdrantEmbeddingStore.builder()
                .host("localhost")
                .port(6334)
                .collectionName("my-collection")
                .build();
         //*/

        //*/ Utilisation d'une base vectorielle en mémoire pour stocker les embeddings et les segments associés.
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        //*

        embeddingStore.addAll(embeddings, segments);

        // Création du ContentRetriever. Nous ne voulons que les 2 résultats
        // les plus pertinents et seulement si leur score est supérieur ou égal à 0,5
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.5)
                .build();

        // Création du WebSearchEngine Tavily
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        // Création du ContentRetriever Web
        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        QueryRouter queryRouter = new DefaultQueryRouter(contentRetriever, webRetriever);

        // RetrievalAugmentor avec le QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Mémoire de 10 messages
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Création de l'assistant. Utilise le modèle Gemini ou le modèle Claude.
        Assistant assistant = AiServices.builder(Assistant.class)
                //.chatModel(modelGemini)
                .chatModel(modelClaude)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Boucle de dialogue avec l'assistant. L'utilisateur pose une question,
        // l'assistant répond en utilisant les informations du document.
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
