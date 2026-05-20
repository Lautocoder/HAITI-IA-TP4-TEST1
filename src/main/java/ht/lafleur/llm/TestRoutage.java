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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import static ht.lafleur.llm.Utils.*;

public class TestRoutage {


    public static void main(String[] args) {
        Utils.configureLogger();

        String pathIADoc = "rag.pdf";
        String pathOtherDoc = "HadoopSparkMapReduce_2.pdf";

        String geminiKey = getKey("GEMINI_KEY");
        String claudeKey= getKey("CLAUDE_KEY");

        if ((geminiKey == null || geminiKey.isBlank()) && (claudeKey == null || claudeKey.isBlank())) {
            System.err.println("Environment variable GEMINI_KEY and CLAUDE_KEY is not set. Set it and retry.");
            System.exit(1);
        }
        // Création des modèles de langage
        ChatModel modelGemini = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-3-flash")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

        ChatModel modelClaude = AnthropicChatModel.builder()
                .apiKey(claudeKey)
                .modelName("claude-sonnet-4-6")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

        // Créez une mémoire pour 10 messages.
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Phase 1 — Ingestion des 2 documents
        EmbeddingStore<TextSegment> storeIADoc = creerEmbeddingStore(pathIADoc);
        EmbeddingStore<TextSegment> storeOtherDoc = creerEmbeddingStore(pathOtherDoc);

        // Phase 2 — ContentRetrievers
        ContentRetriever retrieverIADoc = creerContentRetriever(storeIADoc);
        ContentRetriever retrieverOtherDoc = creerContentRetriever(storeOtherDoc);


        Map<ContentRetriever, String> retrieverDescriptions = new HashMap<>();
        retrieverDescriptions.put(retrieverIADoc,
                "Document sur l'intelligence artificielle, le machine learning, le RAG, le fine-tuning et les LLMs.");
        retrieverDescriptions.put(retrieverOtherDoc,
                "Document sur Hadoop, Spark, MapReduce et les systèmes de fichiers distribués.");

        //  Routage
        QueryRouter routage = new LanguageModelQueryRouter(modelClaude, retrieverDescriptions);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(routage)
                .build();

        // Phase 4 — Test
        Assistant assistant = AiServices.builder(Assistant.class)
                            .chatModel(modelClaude)
                            .retrievalAugmentor(retrievalAugmentor)
                            .chatMemory(chatMemory)
                            .build();


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
