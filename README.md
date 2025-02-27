# RoboExhibit

This repository contains code and resources for RoboExhibit, an AI-powered interactive museum experience. The project integrates intelligent agents, ontology-based knowledge representation, real-time visitor adaptation, and natural language interaction to enhance museum visits dynamically.

## Participants
- **Antonio Spedito**
- **Alessandro Picone**
- **Lucrezia Mosca**

## Project Overview
RoboExhibit is a cutting-edge AI-driven system designed to revolutionize the museum experience by integrating **virtual and physical intelligent agents**. These agents serve as museum guides, dynamically adapting their behavior based on visitor preferences, movement patterns, and environmental conditions. The system leverages **ontology-based knowledge representation, natural language processing (NLP), multi-agent coordination, and real-time navigation algorithms** to create a highly engaging, interactive, and immersive environment tailored to each visitor’s needs.

## Repository Structure
The project is organized into the following key files and directories:

- **Documentazione RoboExhibit.pdf** - Detailed documentation about the project, methodologies, and implementation.
- **Museo.zip** - The full Unity project containing the AI-driven museum experience, including the agent behaviors and interactive environments.
- **Ontologia.rdf** - The RDF ontology used for knowledge representation and semantic queries within the museum.
- **Diagramma delle classi.asta** - A diagram illustrating the class relationships and interactions within the system.
- **video_1.mp4** - A demonstration video showcasing the RoboExhibit experience in action.

## Key Features
- **AI-Powered Multilingual Museum Guides**: RoboExhibit guides provide real-time explanations and recommendations based on visitor interests. They can communicate fluently in multiple languages, ensuring accessibility for international visitors.
- **Ontology-Based Knowledge Representation**: Museum data is structured using an RDF-based knowledge graph, queried dynamically through SPARQL to provide precise and contextual responses to visitor questions.
- **Real-Time Visitor Adaptation and Itinerary Planning**: The system continuously analyzes visitor behavior, dynamically adjusting exhibit routes to optimize engagement and avoid congestion.
- **Navigation and Pathfinding**: Intelligent movement is achieved through a combination of **NavMesh** and **A* pathfinding**, allowing agents to autonomously navigate the museum environment while avoiding obstacles and ensuring smooth visitor experiences.
- **Multi-Agent System Coordination**: RoboExhibit includes various AI-driven agents, such as ticketing assistants, museum guides, and interactive installation managers, all working together to provide a seamless and cohesive experience.
- **Advanced Response Generation Pipeline**: Visitor interactions are processed through a structured **response pipeline** that:
  - Receives natural language questions from visitors.
  - Generates structured **SPARQL queries** to extract relevant knowledge from the ontology.
  - Executes the queries to retrieve museum information.
  - Uses an AI-driven **language model** to generate natural and context-aware responses.
  - Translates responses into multiple languages if necessary.
  - Leverages a **backend hosted on Hugging Face Spaces** to process natural language queries efficiently and generate real-time responses.
- **Human-Robot Collaboration**: The system is designed to complement human staff, allowing for seamless coordination between robotic and human guides, ensuring a rich and flexible experience for visitors.
- **Interactive Digital Exhibits and Multi-Sensory Engagement**: Certain museum rooms feature AI-powered interactive displays that react dynamically to visitor presence and input, offering deeper engagement with artworks and artifacts.
- **Real-Time Replanning and Adaptive Tour Paths**: Unlike static guided tours, RoboExhibit continuously re-evaluates and modifies visitor routes based on preferences, new requests, environmental conditions, and updated museum schedules. This ensures that visitors receive a highly personalized and adaptive tour experience.

## How It Works
1. **Visitor Identification & Profiling**:
   - The system gathers visitor preferences through ticket selection and ongoing interactions.
   - Visitor interests, such as historical or artistic themes, are used to customize the tour dynamically.
2. **Dynamic Itinerary Planning & Replanning**:
   - The AI museum guide plans an optimal path based on visitor preferences and real-time museum conditions.
   - If the visitor requests a change, encounters an obstacle, or the museum environment changes, the system triggers **replanning algorithms** to adjust the itinerary dynamically.
3. **Multilingual Interaction and Knowledge Retrieval**:
   - Visitors can ask questions in various languages, and the guide will retrieve and respond in the same language.
   - The **ontology-based knowledge graph** ensures precise, structured, and context-aware responses.
4. **Advanced Response Generation Pipeline**:
   - The system processes questions using a structured NLP pipeline that involves:
     - Converting questions into structured queries.
     - Extracting knowledge through SPARQL.
     - Refining and formatting responses using an AI-driven language model.
     - Translating responses into the visitor’s language if needed.
     - Utilizing a **backend hosted on Hugging Face Spaces** to process natural language queries and enhance response generation.
5. **Seamless Human-Robot Collaboration**:
   - Both human and robotic guides work together, with robots handling routine explanations and humans assisting with complex queries.
   - The ticketing system is managed by both robotic and human assistants, ensuring flexibility in visitor interactions.
6. **Interactive Installations & Personalized Experience**:
   - Visitors can interact with digital exhibits that respond to presence and gestures.
   - The system can suggest immersive experiences, such as augmented reality tours, to deepen engagement.

## Results
- **Highly Personalized and Adaptive Visitor Experience**: RoboExhibit ensures every visitor receives a tailored tour based on their preferences and language.
- **Efficient Museum Navigation and Crowd Management**: The AI dynamically adjusts routes to optimize flow and engagement.
- **Multi-Agent Collaboration for Seamless Assistance**: Human and robotic guides work together to enhance visitor satisfaction.
- **Advanced NLP and Knowledge Graph Integration**: Ensures accurate, contextual, and multilingual responses to visitor inquiries.
- **Real-Time Replanning and Itinerary Adaptation**: The museum guide continuously adjusts tour paths based on visitor interactions and real-time conditions.
- **Enhanced Accessibility**: By supporting multiple languages and interactive interfaces, RoboExhibit makes museum visits more inclusive.

## Project Report
For more details, refer to the [Documentazione RoboExhibit.pdf](Documentazione_RoboExhibit.pdf) file.

## License
This project is distributed under the license found in the **LICENSE** file.

