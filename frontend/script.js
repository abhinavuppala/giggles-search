
// placeholder results until the search API is connected
const MOCK_RESULTS = [
    { title: "UCI Donald Bren School of Information and Computer Sciences", url: "https://ics.uci.edu/", description: "Information and Computer Sciences at UC Irvine." },
    { title: "UCI Department of Computer Science", url: "https://cs.ics.uci.edu/", description: "Computer science programs and research at UC Irvine." },
    { title: "UCI Department of Informatics", url: "https://www.informatics.uci.edu/", description: "Informatics education and research at UC Irvine." },
    { title: "UCI Center for Machine Learning", url: "https://cml.ics.uci.edu/", description: "Research in machine learning and related areas." },
    { title: "UCI ICS Faculty", url: "https://www.ics.uci.edu/faculty/", description: "Faculty directory for Information and Computer Sciences." }
];

const form = document.getElementById("search-form");
const summarySection = document.getElementById("summary-section");
const resultsSection = document.getElementById("results-section");
const resultsContainer = document.getElementById("search-results");
const stats = document.getElementById("search-stats");

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = form.elements.q.value.trim();
    if (!query) return;

    const limit = event.submitter?.name === "btnI" ? 1 : 5;

    // fetch from our GET /api/search
    // links up to our search engine impl.
    let data;
    try {
        const response = await fetch(
            `/api/search?q=${encodeURIComponent(query)}&limit=${limit}`
        );

        if (!response.ok) {
            throw new Error(`Search failed: ${response.status}`);
        }

        data = await response.json();
    } catch (error) {
        stats.textContent = `Search error: ${error.message}`;
        resultsContainer.replaceChildren();
        resultsSection.hidden = false;
        summarySection.hidden = true;
        return;
    }

    const results = data.results;

    summarySection.hidden = false;
    resultsSection.hidden = false;
    document.getElementById("summary-content").textContent = "";
    stats.textContent = `Found ${data.total} matching documents in ${data.latency_ms.toFixed(4)} ms`;    resultsContainer.replaceChildren();

    for (const result of results.slice(0, limit)) {
        const item = document.createElement("article");
        item.className = "result-item";

        const title = document.createElement("a");
        title.className = "result-title";
        title.href = result.url;
        title.textContent = result.title;

        const url = document.createElement("div");
        url.className = "result-url";
        url.textContent = result.url;

        const description = document.createElement("p");
        description.className = "result-description";
        description.textContent = result.description;

        item.append(title, url, description);
        resultsContainer.appendChild(item);
    }
});
