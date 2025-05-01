export function filterFormData(formData) {
	return Object.fromEntries(
		Array.from(formData.entries()).filter(([key]) => !key.startsWith('$'))
	);
}
