import os
from typing import Dict, Any
import json
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from ..visualization.colors import COLORS
import webbrowser  # Add at the top with other imports

class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        template_dir = os.path.dirname(__file__)
        self.env = Environment(
            loader=FileSystemLoader(template_dir)
        )
        os.makedirs(output_dir, exist_ok=True)

    def _get_css(self) -> str:
        """Load CSS content."""
        css_path = os.path.join(os.path.dirname(__file__), 'styles', 'main.css')
        with open(css_path, 'r') as f:
            return f.read()

    def generate_summary_report(
        self, 
        metrics: Dict[str, float],
        config: Dict[str, Any],
        regional_comparison: Dict[str, Dict[str, float]],
        plots: Dict[str, go.Figure]
    ) -> None:
        """Generate HTML report with visualizations."""
        template = self.env.get_template('templates/report.html')
        
        # Convert plots to JSON with proper escaping
        plot_json = {}
        for name, plot in plots.items():
            try:
                json_str = plot.to_json()
                plot_json[name] = json_str
            except Exception as e:
                logger.error(f"Error converting plot {name} to JSON: {str(e)}")
                plot_json[name] = "{}"
        
        # Get CSS content
        css_content = self._get_css()
        
        # Render template with data
        html_content = template.render(
            metrics=metrics,
            config=config,
            regional_comparison=regional_comparison,
            plots=plot_json,
            css=css_content
        )
        
        # Write report to file
        report_path = os.path.join(self.output_dir, "energy_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        # Open the report in the default browser
        webbrowser.open('file://' + os.path.abspath(report_path)) 