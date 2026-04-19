import React from 'react';
import { motion } from 'framer-motion';

const PipelineStage = ({ number, title, desc, children }) => (
  <motion.div 
    className="pipeline-stage"
    initial={{ opacity: 0, x: -20 }}
    whileInView={{ opacity: 1, x: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
  >
    <div className="stage-number">{number}</div>
    <div className="stage-content">
      <div className="stage-visual">{children}</div>
      <div className="stage-text">
        <h4 className="stage-title">{title}</h4>
        <p className="stage-desc">{desc}</p>
      </div>
    </div>
  </motion.div>
);

const HowWeAnalyze = () => {
  return (
    <div className="analyze-page">
      <header className="analyze-intro">
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          How we <span className="gradient-text">Analyze</span>
        </motion.h1>
        <motion.p 
          className="section-subtitle"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          A step-by-step breakdown of our multi-modal forensic architecture.
        </motion.p>
      </header>

      {/* SigLIP Pipeline */}
      <section className="analyze-section">
        <div className="section-info">
          <span className="model-badge model-siglip">Visual Intelligence</span>
          <h2>SigLIP Pipeline</h2>
          <p>
            The SigLIP (Sigmoid Language-Image Pre-training) process breaks down images into mathematical patches to identify visual manipulation and misinformation patterns.
          </p>
          
          <div className="pipeline-container">
            <PipelineStage 
              number="1" 
              title="Image Resizing" 
              desc="The input image is standardized to 224x224 pixels to ensure consistent processing across all inputs."
            >
              <div style={{ width: '40px', height: '40px', background: 'var(--siglip-gradient)', borderRadius: '4px' }} />
            </PipelineStage>

            <PipelineStage 
              number="2" 
              title="Image Patching (16x16)" 
              desc="The image is divided into a 16x16 grid of patches, allowing the model to focus on granular visual details."
            >
              <div className="mini-grid">
                {[...Array(9)].map((_, i) => <div key={i} className="grid-cell" />)}
              </div>
            </PipelineStage>

            <PipelineStage 
              number="3" 
              title="Patch Flattening & Embedding" 
              desc="Each 2D patch is flattened into a 1D vector and projected into a higher-dimensional embedding space."
            >
              <div style={{ display: 'flex', gap: '2px' }}>
                {[...Array(5)].map((_, i) => <div key={i} style={{ width: '4px', height: '20px', background: 'var(--siglip-gradient)' }} />)}
              </div>
            </PipelineStage>

            <PipelineStage 
              number="4" 
              title="CLS Token & Positional Embeddings" 
              desc="A special [CLS] token is added at the start, and positional information is added to each patch embedding."
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{ padding: '2px 6px', background: '#8b5cf6', borderRadius: '2px', fontSize: '0.6rem' }}>CLS</div>
                <div className="plus-sign" style={{ fontSize: '0.8rem' }}>+</div>
                <div style={{ width: '20px', height: '10px', background: 'var(--surface-1)', borderRadius: '2px' }} />
              </div>
            </PipelineStage>

            <PipelineStage 
              number="5" 
              title="Transformer Encoder Layers" 
              desc="The embeddings pass through multiple layers of Layer Normalization, Multi-Head Self-Attention, and Feed Forward blocks."
            >
              <div className="transformer-block-mini">
                <div className="block-layer" />
                <div className="block-layer" style={{ width: '60%' }} />
                <div className="block-layer" />
              </div>
            </PipelineStage>

            <PipelineStage 
              number="6" 
              title="Classification Verdict" 
              desc="The final CLS vector is passed through a Linear layer and Softmax function to produce the rumor probability."
            >
              <div className="verdict-pulse" style={{ background: 'var(--siglip-gradient)' }} />
            </PipelineStage>
          </div>
        </div>
      </section>

      {/* XLM-RoBERTa Pipeline */}
      <section className="analyze-section">
        <div className="section-info">
          <span className="model-badge model-xlm">Textual Forensics</span>
          <h2>XLM-RoBERTa Pipeline</h2>
          <p>
            Modern news contains textual elements like headlines and descriptions. XLM-RoBERTa analyzes these features across 100+ languages to detect linguistic markers of rumors.
          </p>

          <div className="pipeline-container">
            <PipelineStage 
              number="1" 
              title="Input Features" 
              desc="Headline and description text extracted via OCR are taken as the primary input features."
            >
              <div style={{ fontSize: '1.2rem' }}>📄</div>
            </PipelineStage>

            <PipelineStage 
              number="2" 
              title="Token Embeddings" 
              desc="Text is broken down into sub-word tokens starting with a [CLS] token to represent the whole sentence."
            >
              <div style={{ display: 'flex', gap: '4px' }}>
                <div className="stack-layer layer-token" style={{ width: '20px' }} />
                <div className="stack-layer layer-token" style={{ width: '30px' }} />
              </div>
            </PipelineStage>

            <PipelineStage 
              number="3" 
              title="Embedding Summation" 
              desc="Token, Position, and Segment embeddings are summed together to create the final input vector."
            >
              <div className="embedding-stack">
                <div className="stack-layer layer-token" />
                <div className="stack-layer layer-pos" />
                <div className="stack-layer layer-seg" />
              </div>
            </PipelineStage>

            <PipelineStage 
              number="4" 
              title="XLM-RoBERTa Encoder" 
              desc="The summed embeddings pass through deep bidirectional transformer blocks to capture global context."
            >
              <div className="transformer-block-mini" style={{ borderColor: '#3b82f6' }}>
                <div className="block-layer" style={{ background: '#3b82f6' }} />
                <div className="block-layer" style={{ width: '60%', background: '#3b82f6' }} />
              </div>
            </PipelineStage>

            <PipelineStage 
              number="5" 
              title="News Verdict" 
              desc="The rich contextual features identify whether the textual claims align with rumor patterns or factual reporting."
            >
              <div className="verdict-pulse" style={{ background: 'var(--xlm-gradient)' }} />
            </PipelineStage>
          </div>
        </div>
      </section>

      {/* Gemini & Tavily Section */}
      <section className="analyze-section">
        <div className="section-info">
          <div style={{ display: 'flex', gap: '1rem' }}>
            <span className="model-badge model-gemini">Gemini Intelligence</span>
            <span className="model-badge model-tavily">Tavily Research</span>
          </div>
          <h2>Global Knowledge Synthesis</h2>
          <p>
            When visual and textual features are inconclusive, we use Generative AI and Real-time Web Research to cross-verify claims against the global internet.
          </p>
          
          <div className="pipeline-container">
             <PipelineStage 
              number="1" 
              title="Tavily Web Search" 
              desc="Uses advanced search APIs to gather real-time evidence, official debunks, and factual articles from across the web."
            >
              <div style={{ fontSize: '1.2rem' }}>🌐</div>
            </PipelineStage>
            
            <PipelineStage 
              number="2" 
              title="Gemini Reasoning" 
              desc="Synthesizes all evidence (Visual, Textual, and Search) to form a logical reasoning path for the final verdict."
            >
              <div className="analysis-node" style={{ width: '60px', height: '60px' }}>
                <span style={{ fontSize: '1rem' }}>🧠</span>
              </div>
            </PipelineStage>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HowWeAnalyze;
