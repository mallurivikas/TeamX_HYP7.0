"""
Multimodal Health Scorer
Integrates physiological data, facial expressions, and voice analysis
for comprehensive health assessment
"""
import numpy as np
from typing import Dict, Optional


class MultimodalHealthScorer:
    """
    Enhanced health scoring system that combines:
    1. Physiological data (BMI, BP, glucose, etc.) - from existing ML models
    2. Facial expressions (pain, stress, anxiety) - from FacialAgent
    3. Voice analysis (stress, breathing) - from VoiceAgent (placeholder)
    
    Generates improved composite health score
    """
    
    def __init__(self):
        self.weights = {
            'physiological': 0.60,  # Existing 4 ML models
            'facial': 0.30,         # Face expression analysis
            'voice': 0.10           # Voice analysis (placeholder for now)
        }
    
    def calculate_enhanced_score(
        self,
        physiological_data: Dict,
        facial_data: Optional[Dict] = None,
        voice_data: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate comprehensive health score from multimodal data
        
        Args:
            physiological_data: dict with ML model predictions
                {
                    'heart_risk': 0-100%,
                    'diabetes_risk': 0-100%,
                    'hypertension_risk': 0-100%,
                    'obesity_risk': 0-100%,
                    'composite_risk': 0-100%,
                    'health_score': 0-100
                }
            facial_data: dict with facial analysis (optional)
                {
                    'avg_pain_score': 0-10,
                    'avg_stress_score': 0-10,
                    'avg_anxiety_score': 0-10
                }
            voice_data: dict with voice analysis (optional)
                {
                    'emotion': str,  # happy, sad, neutral, stressed
                    'emotion_confidence': float (0-1),
                    'stress_score': float (0-1),
                    'fatigue_score': float (0-1),
                    'voice_quality': float (0-1),
                    'stability_score': float (0-1),
                    'explanation': str
                }
        
        Returns:
            dict with enhanced health metrics
        """
        
        # Base score from physiological data
        base_health_score = physiological_data.get('health_score', 50.0)
        base_composite_risk = physiological_data.get('composite_risk', 50.0)
        
        # Facial modifiers (if available)
        facial_modifier = 0.0
        if facial_data:
            # Convert facial scores to health impact (0-10 scale)
            pain_impact = facial_data.get('avg_pain_score', 0) * -1.0  # Pain reduces health
            stress_impact = facial_data.get('avg_stress_score', 0) * -0.8
            anxiety_impact = facial_data.get('avg_anxiety_score', 0) * -0.6
            
            facial_modifier = (pain_impact + stress_impact + anxiety_impact) / 3.0
        
        # Voice modifiers (if available) - UPDATED for audio_emotion format
        voice_modifier = 0.0
        if voice_data:
            # Voice data from audio_emotion: stress, fatigue, voice_quality (all 0-1 scale)
            voice_stress = voice_data.get('stress_score', 0)  # 0-1 scale
            voice_fatigue = voice_data.get('fatigue_score', 0)  # 0-1 scale
            voice_quality = voice_data.get('voice_quality', 0.5)  # 0-1 scale
            
            # Convert to 0-10 scale for impact calculation
            # Higher stress/fatigue = worse health
            # Higher quality = better health
            voice_health_impact = (
                -voice_stress * 10 * 0.5 +  # Stress reduces health
                -voice_fatigue * 10 * 0.4 +  # Fatigue reduces health
                (voice_quality * 10 - 5) * 0.3  # Quality above 0.5 improves health
            ) / 3.0
            
            voice_modifier = voice_health_impact
        
        # Calculate enhanced scores
        # Health score: higher is better (0-100)
        enhanced_health_score = base_health_score + (
            facial_modifier * self.weights['facial'] * 10 +
            voice_modifier * self.weights['voice'] * 10
        )
        enhanced_health_score = np.clip(enhanced_health_score, 0, 100)
        
        # Composite risk: lower is better (0-100%)
        facial_risk_increase = 0
        if facial_data:
            avg_facial_score = (
                facial_data.get('avg_pain_score', 0) +
                facial_data.get('avg_stress_score', 0) +
                facial_data.get('avg_anxiety_score', 0)
            ) / 3.0
            facial_risk_increase = (avg_facial_score / 10) * 15  # Max 15% increase
        
        voice_risk_increase = 0
        if voice_data:
            # Calculate risk increase from voice stress and fatigue (0-1 scale)
            voice_stress = voice_data.get('stress_score', 0)
            voice_fatigue = voice_data.get('fatigue_score', 0)
            avg_voice_risk = (voice_stress + voice_fatigue) / 2.0  # 0-1 scale
            voice_risk_increase = avg_voice_risk * 10  # Max 10% increase
        
        enhanced_composite_risk = base_composite_risk + facial_risk_increase + voice_risk_increase
        enhanced_composite_risk = np.clip(enhanced_composite_risk, 0, 100)
        
        # Calculate grade
        grade = self._calculate_grade(enhanced_health_score)
        
        # Generate detailed breakdown
        return {
            'enhanced_health_score': round(enhanced_health_score, 2),
            'enhanced_composite_risk': round(enhanced_composite_risk, 2),
            'grade': grade,
            
            # Modality contributions
            'physiological_contribution': {
                'health_score': base_health_score,
                'composite_risk': base_composite_risk,
                'weight': self.weights['physiological']
            },
            'facial_contribution': {
                'modifier': round(facial_modifier, 2),
                'risk_increase': round(facial_risk_increase, 2),
                'weight': self.weights['facial'],
                'available': facial_data is not None
            },
            'voice_contribution': {
                'modifier': round(voice_modifier, 2),
                'risk_increase': round(voice_risk_increase, 2),
                'weight': self.weights['voice'],
                'available': voice_data is not None
            },
            
            # Detailed metrics
            'individual_risks': physiological_data.get('individual_risks', {}),
            'facial_indicators': facial_data if facial_data else {},
            'voice_indicators': voice_data if voice_data else {},
            
            # Assessment metadata
            'multimodal_complete': all([
                physiological_data is not None,
                facial_data is not None,
                voice_data is not None
            ]),
            'modalities_used': self._count_modalities(physiological_data, facial_data, voice_data)
        }
    
    def _calculate_grade(self, health_score: float) -> str:
        """Calculate letter grade from health score"""
        if health_score >= 90:
            return 'A'
        elif health_score >= 80:
            return 'B'
        elif health_score >= 70:
            return 'C'
        elif health_score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _count_modalities(self, phys, facial, voice):
        """Count how many modalities were used"""
        count = 1  # Physiological always present
        if facial:
            count += 1
        if voice:
            count += 1
        return count
    
    def get_health_insights(self, enhanced_data: Dict) -> str:
        """
        Generate human-readable health insights
        
        Args:
            enhanced_data: Output from calculate_enhanced_score()
            
        Returns:
            Formatted insights string
        """
        insights = []
        
        score = enhanced_data['enhanced_health_score']
        risk = enhanced_data['enhanced_composite_risk']
        grade = enhanced_data['grade']
        
        # Overall assessment
        if score >= 80:
            insights.append(f"✅ Excellent health status (Grade {grade})")
        elif score >= 60:
            insights.append(f"⚠️ Good health with room for improvement (Grade {grade})")
        else:
            insights.append(f"🔴 Health concerns detected - consult healthcare provider (Grade {grade})")
        
        # Facial indicators
        if enhanced_data['facial_contribution']['available']:
            facial_data = enhanced_data['facial_indicators']
            pain = facial_data.get('avg_pain_score', 0)
            stress = facial_data.get('avg_stress_score', 0)
            
            if pain > 5:
                insights.append(f"😟 Significant pain indicators detected (Pain Score: {pain:.1f}/10)")
            if stress > 5:
                insights.append(f"😰 Elevated stress levels observed (Stress Score: {stress:.1f}/10)")
        
        # Voice indicators - UPDATED for audio_emotion
        if enhanced_data['voice_contribution']['available']:
            voice_data = enhanced_data['voice_indicators']
            emotion = voice_data.get('emotion', 'neutral')
            stress = voice_data.get('stress_score', 0) * 10  # Convert to 0-10 scale
            fatigue = voice_data.get('fatigue_score', 0) * 10  # Convert to 0-10 scale
            
            if stress > 6:
                insights.append(f"🎤 High voice stress detected (Stress: {stress:.1f}/10)")
            if fatigue > 6:
                insights.append(f"😴 Voice fatigue indicators present (Fatigue: {fatigue:.1f}/10)")
            if emotion == 'stressed':
                insights.append(f"😰 Stressed emotional state detected in voice")
            elif emotion == 'sad':
                insights.append(f"😢 Low mood detected in voice patterns")
        
        # Risk assessment
        if risk < 30:
            insights.append(f"✅ Low overall health risk ({risk:.1f}%)")
        elif risk < 60:
            insights.append(f"⚠️ Moderate health risk - lifestyle changes recommended ({risk:.1f}%)")
        else:
            insights.append(f"🔴 High health risk - medical consultation advised ({risk:.1f}%)")
        
        return "\n".join(insights)


# Convenience function
def calculate_multimodal_health(physiological, facial=None, voice=None):
    """
    Calculate enhanced health score from multimodal data
    
    Args:
        physiological: ML model predictions
        facial: Facial analysis results (optional)
        voice: Voice analysis results (optional)
        
    Returns:
        Enhanced health assessment dict
    """
    scorer = MultimodalHealthScorer()
    return scorer.calculate_enhanced_score(physiological, facial, voice)
