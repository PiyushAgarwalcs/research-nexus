import logging
import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


def _normalize_cross_encoder_score(score: float) -> float:
    """Squash a raw cross-encoder logit into (0, 1) via a sigmoid."""
    return 1.0 / (1.0 + math.exp(-score))


class EvidenceStance(str, Enum):
    """Scientific stance of retrieved evidence towards a query or claim."""
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"


class PassageStance(BaseModel):
    chunk_id: int = Field(description="1-based index of the passage")
    stance: str = Field(
        description="SUPPORT, CONTRADICT, NEUTRAL, or UNCERTAIN towards the research question or claim"
    )
    key_finding: str = Field(description="The scientific finding or claim made in this passage")
    methodology: Optional[str] = Field(default=None, description="Research methodology or experimental setup")
    sample_or_conditions: Optional[str] = Field(default=None, description="Sample size, population, or trial conditions")
    limitations: Optional[str] = Field(default=None, description="Limitations, constraints, or caveats mentioned")
    reasoning: str = Field(description="Why this passage was assigned this stance")


class EvidenceAnalysisReport(BaseModel):
    findings: List[PassageStance] = Field(default_factory=list)
    consensus_summary: str = Field(default="Evidence analysis completed.")
    evidence_gaps: List[str] = Field(default_factory=list, description="Aspects missing or inconclusive")


class CitationVerification(BaseModel):
    claim: str = Field(description="The specific claim extracted from the answer")
    cited_chunk_ids: List[int] = Field(
        description="The IDs of the chunks cited for this claim"
    )
    is_supported: bool = Field(
        description="Whether the source text fully supports the claim"
    )
    reasoning: str = Field(
        description="Explanation of why the claim is or is not supported"
    )


class VerificationResult(BaseModel):
    verifications: List[CitationVerification]
    coverage_percentage: float


class ConfidenceScore(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite_score: float
    is_confident: bool


class AdvancedRAGSystem:
    def __init__(self, llm, retriever, confidence_threshold=settings.confidence_threshold):
        self.llm = llm
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        try:
            self.judge_llm = llm.with_structured_output(CitationVerification)
            self.evidence_judge_llm = llm.with_structured_output(EvidenceAnalysisReport)
        except Exception:
            self.judge_llm = None
            self.evidence_judge_llm = None

    def parse_citations(self, answer: str) -> List[Dict[str, Any]]:
        """Splits an answer into (claim, cited chunk ids) pairs. Back-to-back
        citations on the same claim (e.g. "...the Moon [1][2].") attach ALL
        of their chunk ids to that one claim, rather than the second (and
        any subsequent) citation being silently dropped because there's no
        text between the brackets to anchor a new claim to."""
        claims = []
        # Matches "[1]" as well as the comma-separated form "[1, 2]".
        citation = r"\[\d+(?:\s*,\s*\d+)*\]"
        parts = re.split(f"({citation})", answer)
        current_claim = ""

        for part in parts:
            if re.fullmatch(citation, part):
                chunk_ids = [int(n) for n in re.findall(r"\d+", part)]
                if current_claim.strip():
                    claims.append(
                        {"claim": current_claim.strip(), "chunk_ids": chunk_ids}
                    )
                elif claims:
                    claims[-1]["chunk_ids"].extend(chunk_ids)
                current_claim = ""
            else:
                current_claim += part

        return claims

    def verify_citations(
        self, claims: List[Dict[str, Any]], retrieved_chunks: List[Any]
    ) -> VerificationResult:
        verifications = []
        supported_count = 0

        for claim_data in claims:
            chunk_ids = claim_data["chunk_ids"]
            claim_text = claim_data["claim"]

            chunk_contents = [
                retrieved_chunks[cid - 1].page_content
                for cid in chunk_ids
                if 0 < cid <= len(retrieved_chunks)
            ]
            source_text = "\n\n".join(chunk_contents)

            prompt = f"""
            Evaluate if the following claim is fully supported by the provided source
            text (which may combine multiple cited chunks).
            Claim: {claim_text}
            Cited Chunk IDs: {chunk_ids}
            Source Text: {source_text}
            """

            try:
                result = self.judge_llm.invoke(prompt)
            except Exception as e:
                logger.warning("Structured citation verification failed: %s; falling back to heuristic", e)
                # Fallback: check if key tokens from claim appear in source
                tokens = [t.lower() for t in claim_text.split() if len(t) > 4]
                src_lower = source_text.lower()
                matched = sum(1 for t in tokens if t in src_lower)
                is_sup = (matched / len(tokens)) > 0.5 if tokens else True
                result = CitationVerification(
                    claim=claim_text,
                    cited_chunk_ids=chunk_ids,
                    is_supported=is_sup,
                    reasoning="Heuristic keyword alignment verification",
                )

            verifications.append(result)
            if result.is_supported:
                supported_count += 1

        coverage = (supported_count / len(claims)) if claims else 1.0

        return VerificationResult(
            verifications=verifications, coverage_percentage=coverage
        )

    def analyze_scientific_evidence(
        self, query: str, retrieved_chunks: List[Any], raw_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze scientific stance (SUPPORT, CONTRADICT, NEUTRAL, UNCERTAIN)
        and extract findings, methodologies, and limitations for each passage.
        Never confuses retrieval relevance with scientific agreement."""
        if not retrieved_chunks:
            return {
                "findings": [],
                "supporting_count": 0,
                "contradicting_count": 0,
                "neutral_count": 0,
                "uncertain_count": 0,
                "consensus_summary": "No evidence retrieved.",
                "evidence_gaps": ["No relevant scientific documents found."],
            }

        # Build prompt for structured analysis
        passage_blocks = "\n\n".join(
            [
                f"[Passage {i+1}]:\n{c.page_content}"
                for i, c in enumerate(retrieved_chunks)
            ]
        )

        prompt = f"""
Analyze the retrieved scientific passages with respect to the research query.
Query: {query}

Passages:
{passage_blocks}

Instructions:
1. For EACH passage, evaluate whether its scientific evidence SUPPORTS, CONTRADICTS, is NEUTRAL towards, or UNCERTAIN regarding the query.
2. Note: Retrieval relevance does NOT imply scientific support. A highly relevant paper may refute a claim.
3. Extract specific findings, methodology, and limitations where stated.
4. When evidence is ambiguous or inconclusive, label as UNCERTAIN.
"""
        try:
            if hasattr(self, "evidence_judge_llm") and self.evidence_judge_llm:
                report: EvidenceAnalysisReport = self.evidence_judge_llm.invoke(prompt)
                findings = [f.model_dump() for f in report.findings]
                consensus = report.consensus_summary
                gaps = report.evidence_gaps
            else:
                findings = self._heuristic_evidence_analysis(query, retrieved_chunks)
                consensus = "Heuristic analysis based on retrieved scientific context."
                gaps = []
        except Exception as e:
            logger.warning("Evidence analysis structured call failed: %s; using heuristic", e)
            findings = self._heuristic_evidence_analysis(query, retrieved_chunks)
            consensus = "Scientific analysis generated from retrieved context."
            gaps = []

        sup = sum(1 for f in findings if f.get("stance") == "SUPPORT")
        con = sum(1 for f in findings if f.get("stance") == "CONTRADICT")
        neu = sum(1 for f in findings if f.get("stance") == "NEUTRAL")
        unc = sum(1 for f in findings if f.get("stance") == "UNCERTAIN")

        return {
            "findings": findings,
            "supporting_count": sup,
            "contradicting_count": con,
            "neutral_count": neu,
            "uncertain_count": unc,
            "consensus_summary": consensus,
            "evidence_gaps": gaps,
        }

    def _heuristic_evidence_analysis(self, query: str, retrieved_chunks: List[Any]) -> List[Dict[str, Any]]:
        """Fallback rule-based scientific stance analyzer when LLM structured output is unavailable."""
        results = []
        contradict_words = {"not", "fails", "no evidence", "ineffective", "disproved", "contrary", "refute", "unsupported", "negative"}
        support_words = {"demonstrated", "confirmed", "effective", "proves", "increased", "benefit", "improved", "validated", "evidence shows"}

        for i, chunk in enumerate(retrieved_chunks, start=1):
            text_lower = chunk.page_content.lower()
            con_hits = sum(1 for w in contradict_words if w in text_lower)
            sup_hits = sum(1 for w in support_words if w in text_lower)

            if con_hits > sup_hits:
                stance = "CONTRADICT"
                reason = "Passage contains negative/contradictory outcome indicators."
            elif sup_hits > con_hits:
                stance = "SUPPORT"
                reason = "Passage reports positive validation or confirmatory findings."
            elif con_hits == 0 and sup_hits == 0:
                stance = "NEUTRAL"
                reason = "Passage provides descriptive context without a definitive stance."
            else:
                stance = "UNCERTAIN"
                reason = "Passage presents mixed or preliminary evidence."

            first_sentence = chunk.page_content.split(".")[0].strip()[:200]
            results.append({
                "chunk_id": i,
                "stance": stance,
                "key_finding": first_sentence or "Document section",
                "methodology": "Empirical observation" if "study" in text_lower or "trial" in text_lower else None,
                "sample_or_conditions": None,
                "limitations": "Observational study limitations" if "limit" in text_lower else None,
                "reasoning": reason,
            })
        return results

    def score_confidence(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Any],
        coverage: float,
        rerank_scores: Optional[List[float]] = None,
    ) -> ConfidenceScore:
        top_scores = (rerank_scores or [])[:3]
        retrieval_score = (
            sum(_normalize_cross_encoder_score(s) for s in top_scores) / len(top_scores)
            if top_scores
            else 0.0
        )

        completeness_prompt = f"""
        Rate how completely the answer addresses the query on a scale of 0.0 to 1.0.
        Query: {query}
        Answer: {answer}
        Output ONLY a float value, nothing else.
        """
        try:
            completeness_result = float(
                self.llm.invoke(completeness_prompt).content.strip()
            )
        except Exception:
            completeness_result = 0.5
        if not math.isfinite(completeness_result):
            completeness_result = 0.5
        completeness_result = min(max(completeness_result, 0.0), 1.0)

        composite = (
            (retrieval_score * 0.3) + (coverage * 0.4) + (completeness_result * 0.3)
        )

        return ConfidenceScore(
            retrieval_confidence=retrieval_score,
            citation_coverage=coverage,
            answer_completeness=completeness_result,
            composite_score=composite,
            is_confident=composite >= self.confidence_threshold,
        )

    def generate_robust_answer(self, query: str):
        ranked_chunks = self.retriever.get_relevant_documents(query)

        if not ranked_chunks:
            return self._format_unknown_response(query, "No documents retrieved.")

        chunks = [doc for doc, _ in ranked_chunks]
        rerank_scores = [score for _, score in ranked_chunks]

        context = "\n".join(
            [
                f"Context Block [{i + 1}]:\n{c.page_content}"
                for i, c in enumerate(chunks)
            ]
        )

        qa_prompt = f"""
        You are a precise scientific research assistant. Answer the query using ONLY the provided context blocks.
        Cite specific chunks using bracketed references (e.g., [1]).
        Context:
        {context}
        Query: {query}
        """
        raw_answer = self.llm.invoke(qa_prompt).content

        claims = self.parse_citations(raw_answer)
        verification = self.verify_citations(claims, chunks)
        confidence = self.score_confidence(
            query, raw_answer, chunks, verification.coverage_percentage, rerank_scores
        )

        evidence_analysis = self.analyze_scientific_evidence(query, chunks, raw_answer)

        model_info = {
            "name": getattr(self.retriever, "model_target", "ms-marco-MiniLM-L-6-v2"),
            "label": getattr(self.retriever, "model_label", "Cross-Encoder Reranker"),
        }

        if not confidence.is_confident:
            return self._format_unknown_response(
                query,
                "System confidence fell below threshold.",
                chunks,
                confidence.composite_score,
                active_model=model_info,
            )

        return {
            "status": "Success",
            "answer": raw_answer,
            "retrieved_chunks": chunks,
            "confidence_metrics": confidence.model_dump(),
            "flagged_citations": [
                v.model_dump() for v in verification.verifications if not v.is_supported
            ],
            "verified_citations": [
                v.model_dump() for v in verification.verifications if v.is_supported
            ],
            "evidence_analysis": evidence_analysis,
            "active_model": model_info,
        }

    def _format_unknown_response(
        self,
        query: str,
        reason: str,
        chunks: Optional[List[Any]] = None,
        score: float = 0.0,
        active_model: Optional[Dict[str, str]] = None,
    ):
        retriever_obj = getattr(self, "retriever", None)
        default_model = {
            "name": getattr(retriever_obj, "model_target", "unknown") if retriever_obj else "unknown",
            "label": getattr(retriever_obj, "model_label", "Unknown Reranker") if retriever_obj else "Unknown Reranker",
        }

        response = {
            "status": "Insufficient Information",
            "reason": reason,
            "confidence_score": score,
            "found_context": "The system retrieved some potentially related information but could not formulate a reliable answer.",
            "missing_information": query,
            "recommended_action": "Review the suggested documents manually or provide additional scientific papers.",
            "active_model": active_model or default_model,
        }

        if chunks:
            response["suggested_documents"] = list(
                dict.fromkeys(
                    c.metadata.get("filepath", "Unknown source") for c in chunks[:3]
                )
            )

        return response
