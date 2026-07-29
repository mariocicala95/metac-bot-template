import argparse
import asyncio
import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class SummerTemplateBot2026(ForecastBot):
    """
    Ensemble-ready bot for Summer 2026 Metaculus AI Tournament.

    TPM-aware design (Groq free tier limits: 70b=12k, 8b=6k):
    - Forecast ensemble: llama-3.3-70b only by default (~2k TPM/question on 70b).
      Adds Claude Haiku when ANTHROPIC_API_KEY is set (separate TPM pool → true
      multi-model ensemble with no Groq budget competition).
    - Research: AskNews → SmartSearcher → 8b LLM fallback. The 8b is used here
      (not 70b) so the 70b budget is preserved for forecasting.
    - Parsing: 8b model. Combined 8b usage (research fallback + parsing) stays
      under ~2k TPM/question, well within the 6k limit.
    - All questions are processed sequentially (_max_concurrent_questions=1) to
      avoid bursty parallel usage that would trip per-minute rate limits.
    - Partial failures (rate limit on some questions) are tolerated; only a total
      wipeout (0 forecasts) raises RuntimeError and fails the workflow.

    Pending credentials (add as GitHub Secrets to unlock improvements):
    - ASKNEWS_CLIENT_ID + ASKNEWS_SECRET → real-time news research (Free Tier)
      Signup: https://my.asknews.app
    - ANTHROPIC_API_KEY → activates 2-model ensemble (70b + Haiku)
    - EXA_API_KEY or PERPLEXITY_API_KEY → enables SmartSearcher web research
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    # 1 sample is enough: halves parser calls vs. the default 2, which matters
    # because Groq free-tier TPM limits are low (6–12k/min per model).
    _structure_output_validation_samples = 1

    # Primary Groq model: best quality, 12k TPM on free tier.
    _GROQ_PRIMARY = "groq/llama-3.3-70b-versatile"
    # Secondary Groq model: fast, separate TPM pool (6k TPM free tier).
    _GROQ_SECONDARY = "groq/llama-3.1-8b-instant"
    # Anthropic model (used when ANTHROPIC_API_KEY is set).
    _ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5-20251001"

    def _get_ensemble_models(self) -> list[GeneralLlm]:
        """Return LLMs for ensemble based on available credentials.

        Free tier (Groq only): single llama-3.3-70b model. This uses at most
        ~2k TPM/question, safely within the 12k free-tier limit even for a
        batch of test questions. The 8b model is reserved for research fallback
        and parsing — adding it here would exhaust its 6k TPM limit immediately.

        With ANTHROPIC_API_KEY: adds Claude Haiku (separate TPM pool → true
        2-model ensemble with no Groq competition).
        """
        models: list[GeneralLlm] = [
            GeneralLlm(
                model=self._GROQ_PRIMARY,
                temperature=0.3,
                timeout=60,
                allowed_tries=3,
            ),
        ]
        if os.getenv("ANTHROPIC_API_KEY"):
            models.append(
                GeneralLlm(
                    model=self._ANTHROPIC_MODEL,
                    temperature=0.3,
                    timeout=60,
                    allowed_tries=2,
                )
            )
        return models

    def _primary_llm(self, timeout: int = 120, allowed_tries: int = 4) -> GeneralLlm:
        """High-retry primary LLM for last-resort fallback."""
        return GeneralLlm(
            model=self._GROQ_PRIMARY,
            temperature=0.3,
            timeout=timeout,
            allowed_tries=allowed_tries,
        )

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            # Priority 1: AskNews — real-time news search (best for current events).
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                try:
                    research = await AskNewsSearcher().call_preconfigured_version(
                        "asknews/news-summaries", prompt
                    )
                    logger.info(f"AskNews research for {question.page_url}: done")
                    return research
                except Exception as e:
                    logger.warning(f"AskNews failed, falling through: {e}")

            # Priority 2: SmartSearcher — web search with LLM synthesis.
            if os.getenv("EXA_API_KEY") or os.getenv("PERPLEXITY_API_KEY"):
                try:
                    searcher = SmartSearcher(
                        model=self._GROQ_PRIMARY,
                        temperature=0,
                        num_searches_to_run=2,
                        num_sites_per_search=10,
                        use_advanced_filters=False,
                    )
                    research = await searcher.invoke(prompt)
                    logger.info(f"SmartSearcher research for {question.page_url}: done")
                    return research
                except Exception as e:
                    logger.warning(f"SmartSearcher failed, falling through: {e}")

            # Fallback: LLM knowledge only. Use 8b to preserve 70b TPM for
            # forecasting — research quality difference is acceptable here.
            researcher_llm = GeneralLlm(
                model=self._GROQ_SECONDARY, temperature=0.3, timeout=60, allowed_tries=2
            )
            research = await researcher_llm.invoke(prompt)
            logger.info(f"LLM-only research for {question.page_url}: done")
            return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        async with self._concurrency_limiter:
            ensemble_models = self._get_ensemble_models()
            probabilities: list[float] = []
            reasonings: list[str] = []

            for llm in ensemble_models:
                try:
                    reasoning = await llm.invoke(prompt)
                    binary_prediction: BinaryPrediction = await structure_output(
                        reasoning,
                        BinaryPrediction,
                        model=self.get_llm("parser", "llm"),
                        num_validation_samples=self._structure_output_validation_samples,
                    )
                    p = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
                    probabilities.append(p)
                    reasonings.append(f"### [{llm.model}]\n{reasoning}")
                except Exception as e:
                    logger.warning(
                        f"Ensemble model {llm.model} failed on {question.page_url}: {e}"
                    )

            if not probabilities:
                # All ensemble models failed — retry with primary alone at higher persistence.
                logger.warning(
                    f"All ensemble models failed for {question.page_url}; "
                    f"retrying with primary model (allowed_tries=4)."
                )
                reasoning = await self._primary_llm().invoke(prompt)
                binary_prediction = await structure_output(
                    reasoning,
                    BinaryPrediction,
                    model=self.get_llm("parser", "llm"),
                    num_validation_samples=1,
                )
                p = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
                return ReasonedPrediction(prediction_value=p, reasoning=reasoning)

            # Aggregate: arithmetic mean across successful models.
            final_prob = max(0.01, min(0.99, statistics.mean(probabilities)))
            logger.info(
                f"Ensemble binary [{question.page_url}]: "
                f"{[f'{p:.3f}' for p in probabilities]} → {final_prob:.3f}"
            )
            return ReasonedPrediction(
                prediction_value=final_prob,
                reasoning="\n\n---\n\n".join(reasonings),
            )

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )

        async with self._concurrency_limiter:
            ensemble_models = self._get_ensemble_models()
            all_option_probs: list[dict[str, float]] = []
            reasonings: list[str] = []
            last_fallback: PredictedOptionList | None = None

            for llm in ensemble_models:
                try:
                    reasoning = await llm.invoke(prompt)
                    predicted_option_list: PredictedOptionList = await structure_output(
                        text_to_structure=reasoning,
                        output_type=PredictedOptionList,
                        model=self.get_llm("parser", "llm"),
                        num_validation_samples=self._structure_output_validation_samples,
                        additional_instructions=parsing_instructions,
                    )
                    option_probs = {
                        opt.option_name: opt.probability
                        for opt in predicted_option_list.predicted_options
                    }
                    all_option_probs.append(option_probs)
                    reasonings.append(f"### [{llm.model}]\n{reasoning}")
                    if last_fallback is None:
                        last_fallback = predicted_option_list
                except Exception as e:
                    logger.warning(
                        f"Ensemble model {llm.model} failed on {question.page_url}: {e}"
                    )

            if not all_option_probs:
                # All ensemble models failed — retry with primary alone.
                logger.warning(
                    f"MC ensemble fully failed for {question.page_url}; "
                    f"retrying with primary model (allowed_tries=4)."
                )
                reasoning = await self._primary_llm().invoke(prompt)
                fallback_list: PredictedOptionList = await structure_output(
                    text_to_structure=reasoning,
                    output_type=PredictedOptionList,
                    model=self.get_llm("parser", "llm"),
                    num_validation_samples=1,
                    additional_instructions=parsing_instructions,
                )
                return ReasonedPrediction(prediction_value=fallback_list, reasoning=reasoning)

            # Average each option's probability across all models, then renormalize.
            avg_probs: dict[str, float] = {}
            for option in question.options:
                values = [d.get(option, 0.0) for d in all_option_probs]
                avg_probs[option] = statistics.mean(values)

            total = sum(avg_probs.values()) or 1.0
            normalized = {k: v / total for k, v in avg_probs.items()}

            # Re-parse the averaged probabilities into the required type.
            averaged_text = "\n".join(
                f"{opt}: {prob:.4f}" for opt, prob in normalized.items()
            )
            final_option_list: PredictedOptionList = await structure_output(
                text_to_structure=averaged_text,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=1,
                additional_instructions=parsing_instructions,
            )

            logger.info(f"Ensemble MC [{question.page_url}]: {normalized}")
            return ReasonedPrediction(
                prediction_value=final_option_list,
                reasoning="\n\n---\n\n".join(reasonings),
            )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )

        async with self._concurrency_limiter:
            ensemble_models = self._get_ensemble_models()
            all_percentile_lists: list[list[Percentile]] = []
            reasonings: list[str] = []

            for llm in ensemble_models:
                try:
                    reasoning = await llm.invoke(prompt)
                    percentile_list: list[Percentile] = await structure_output(
                        reasoning,
                        list[Percentile],
                        model=self.get_llm("parser", "llm"),
                        additional_instructions=parsing_instructions,
                        num_validation_samples=self._structure_output_validation_samples,
                    )
                    if percentile_list:
                        all_percentile_lists.append(percentile_list)
                        reasonings.append(f"### [{llm.model}]\n{reasoning}")
                except Exception as e:
                    logger.warning(
                        f"Ensemble model {llm.model} failed on {question.page_url}: {e}"
                    )

            if not all_percentile_lists:
                # All ensemble models failed — retry with primary alone.
                logger.warning(
                    f"All ensemble models failed for {question.page_url}; "
                    f"retrying with primary model (allowed_tries=4)."
                )
                reasoning = await self._primary_llm().invoke(prompt)
                percentile_list = await structure_output(
                    reasoning,
                    list[Percentile],
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=1,
                )
                prediction = NumericDistribution.from_question(percentile_list, question)
                return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

            # Average percentile values across models, matching on percentile number.
            percentile_buckets: dict[float, list[float]] = {}
            for plist in all_percentile_lists:
                for p in plist:
                    percentile_buckets.setdefault(p.percentile, []).append(p.value)

            averaged_percentiles = [
                Percentile(percentile=pct, value=statistics.mean(vals))
                for pct, vals in sorted(percentile_buckets.items())
            ]

            prediction = NumericDistribution.from_question(averaged_percentiles, question)
            logger.info(
                f"Ensemble numeric [{question.page_url}]: "
                f"{[(p.percentile, round(p.value, 3)) for p in averaged_percentiles]}"
            )
            return ReasonedPrediction(
                prediction_value=prediction,
                reasoning="\n\n---\n\n".join(reasonings),
            )

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        return await self._date_prompt_to_forecast(question, prompt)

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )

        async with self._concurrency_limiter:
            ensemble_models = self._get_ensemble_models()
            all_percentile_lists: list[list[Percentile]] = []
            reasonings: list[str] = []

            for llm in ensemble_models:
                try:
                    reasoning = await llm.invoke(prompt)
                    date_percentile_list: list[DatePercentile] = await structure_output(
                        reasoning,
                        list[DatePercentile],
                        model=self.get_llm("parser", "llm"),
                        additional_instructions=parsing_instructions,
                        num_validation_samples=self._structure_output_validation_samples,
                    )
                    if date_percentile_list:
                        # Convert datetime → Unix timestamp for numeric averaging.
                        as_timestamps = [
                            Percentile(
                                percentile=dp.percentile,
                                value=dp.value.timestamp(),
                            )
                            for dp in date_percentile_list
                        ]
                        all_percentile_lists.append(as_timestamps)
                        reasonings.append(f"### [{llm.model}]\n{reasoning}")
                except Exception as e:
                    logger.warning(
                        f"Ensemble model {llm.model} failed on {question.page_url}: {e}"
                    )

            if not all_percentile_lists:
                # All ensemble models failed — retry with primary alone.
                logger.warning(
                    f"All ensemble models failed for {question.page_url}; "
                    f"retrying with primary model (allowed_tries=4)."
                )
                reasoning = await self._primary_llm().invoke(prompt)
                date_percentile_list = await structure_output(
                    reasoning,
                    list[DatePercentile],
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                    num_validation_samples=1,
                )
                percentile_list = [
                    Percentile(percentile=dp.percentile, value=dp.value.timestamp())
                    for dp in date_percentile_list
                ]
                prediction = NumericDistribution.from_question(percentile_list, question)
                return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

            # Average timestamp values per percentile bucket.
            percentile_buckets: dict[float, list[float]] = {}
            for plist in all_percentile_lists:
                for p in plist:
                    percentile_buckets.setdefault(p.percentile, []).append(p.value)

            averaged_percentiles = [
                Percentile(percentile=pct, value=statistics.mean(vals))
                for pct, vals in sorted(percentile_buckets.items())
            ]

            prediction = NumericDistribution.from_question(averaged_percentiles, question)
            logger.info(
                f"Ensemble date [{question.page_url}]: "
                f"{[(p.percentile, p.value) for p in averaged_percentiles]}"
            )
            return ReasonedPrediction(
                prediction_value=prediction,
                reasoning="\n\n---\n\n".join(reasonings),
            )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def log_report_summary(self, forecast_reports: list) -> None:
        """Override SDK default: only raise if ALL forecasts failed.
        Partial failure (e.g., transient rate limits on some questions) is
        acceptable for a cron bot — log errors, keep exit code 0 when any
        question succeeded so GitHub Actions doesn't mark the run as broken."""
        from forecasting_tools import ForecastReport

        valid = [r for r in forecast_reports if isinstance(r, ForecastReport)]
        errors = [r for r in forecast_reports if isinstance(r, BaseException)]

        if errors:
            for err in errors:
                logger.error(f"Forecast failed: {err}")
            logger.warning(
                f"{len(errors)}/{len(forecast_reports)} questions failed "
                f"(partial run — see errors above)."
            )
        if valid:
            logger.info(
                f"Successfully forecasted {len(valid)}/{len(forecast_reports)} questions."
            )
        if not valid:
            raise RuntimeError(
                f"All {len(forecast_reports)} forecasts failed — check logs."
            )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # llms= configures parser and summarizer only — forecasting and research are
    # handled by the overridden methods above. Both use the 8b model (light tasks)
    # so the 70b's 12k TPM is fully available for forecasting.
    #
    # Ensemble (from _get_ensemble_models):
    #   GROQ only → single llama-3.3-70b-versatile (~2k TPM/question)
    #   + ANTHROPIC_API_KEY → adds claude-haiku-4-5 (true 2-model ensemble)
    #
    # Research (first available wins):
    #   ASKNEWS_CLIENT_ID + ASKNEWS_SECRET → AskNews real-time news
    #   EXA_API_KEY or PERPLEXITY_API_KEY  → SmartSearcher web search
    #   (fallback)                          → llama-3.1-8b-instant LLM knowledge
    template_bot = SummerTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            # default is bypassed by ensemble overrides but kept as fallback.
            "default": GeneralLlm(
                model="groq/llama-3.3-70b-versatile",
                temperature=0.3,
                timeout=60,
                allowed_tries=2,
            ),
            # parser and summarizer use the 8b model (parsing/summarising don't
            # need the 70b and using 8b preserves the 70b's 12k TPM).
            "summarizer": GeneralLlm(
                model="groq/llama-3.1-8b-instant",
                temperature=0.3,
                timeout=60,
                allowed_tries=3,
            ),
            "parser": GeneralLlm(
                model="groq/llama-3.1-8b-instant",
                temperature=0.3,
                timeout=60,
                allowed_tries=3,
            ),
            "researcher": GeneralLlm(
                model="groq/llama-3.3-70b-versatile",
                temperature=0.3,
                timeout=60,
                allowed_tries=2,
            ),
        },
    )

    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/summer-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-summer-2025/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
