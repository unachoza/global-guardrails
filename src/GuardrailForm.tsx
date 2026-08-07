import { useState, type FormEvent } from "react";

const MAX_LENGTH = 200;

type GuardrailFormProps = {
	onSubmit: (guardrail: string) => void;
};

function GuardrailForm({ onSubmit }: GuardrailFormProps) {
	const [value, setValue] = useState("");

	const trimmed = value.trim();
	const isEmpty = trimmed === "";

	const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
		event.preventDefault();
		if (isEmpty) return;
		onSubmit(trimmed);
		setValue("");
	};

	return (
		<form className="guardrail-form" onSubmit={handleSubmit}>
			<input
				type="text"
				className="guardrail-input"
				aria-label="what guardrails matter to you?"
				placeholder="what guardrails matter to you?"
				maxLength={MAX_LENGTH}
				value={value}
				onChange={(event) => setValue(event.target.value)}
			/>
			<button type="submit" className="counter" disabled={isEmpty}>
				Enter
			</button>
		</form>
	);
}

export default GuardrailForm;
