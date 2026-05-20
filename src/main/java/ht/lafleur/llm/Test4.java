package ht.lafleur.llm;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.util.*;

import static ht.lafleur.llm.Utils.creerContentRetriever;
import static ht.lafleur.llm.Utils.creerEmbeddingStore;

public class Test4 {

    static EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

    public static void main(String[] args) {
        Utils.configureLogger();

        String claudeKey= Utils.getKey("CLAUDE_KEY");

        ChatModel model = AnthropicChatModel.builder()
                .apiKey(claudeKey)
                .modelName("claude-sonnet-4-6")
                .temperature(0.0)  // 0 pour des réponses oui/non stables
                .logRequests(true)
                .logResponses(true)
                .build();

        // Phase 1 — Ingestion du document RAG
        EmbeddingStore<TextSegment> storeIA = creerEmbeddingStore("rag.pdf");

        // Phase 2 — ContentRetriever
        ContentRetriever retrieverIA = creerContentRetriever(storeIA);

        // Template de prompt pour demander au LM si la question porte sur l'IA
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{question}}' porte sur l'IA, " +
                        "le machine learning, les LLMs, le RAG ou le fine-tuning ? " +
                        "Réponds seulement par 'oui', 'non' ou 'peut-être'."
        );

        // QueryRouter personnalisé — classe interne anonyme
        QueryRouter queryRouter = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                String question = query.text();

                // On demande au LM si la question porte sur l'IA
                String prompt = promptTemplate.apply(Map.of("question", question)).text();
                String reponse = model.chat(prompt).trim().toLowerCase();


                System.out.println("[QueryRouter] → RAG activé.");
                return List.of(retrieverIA);
            }
        };

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Boucle de dialogue
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question (ou 'fin' pour quitter) : ");
                String question = scanner.nextLine();
                if (question.isBlank()) continue;
                if ("fin".equalsIgnoreCase(question)) break;
                System.out.println("==================================================");
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
