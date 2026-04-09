"""
Seed script for populating audit_logs and content tables with test data.

Matches the exact database schema (BaseEntity, enums, junction tables).

Usage:
    python -m data.seed_data                  # seed both databases
    python -m data.seed_data --only audit     # seed only audit_logs
    python -m data.seed_data --only content   # seed only content tables
    python -m data.seed_data --clean          # drop & recreate before seeding

Tables seeded:
  [Content DB]  content, category, tag, actor, director,
                movies, tvseries, season, episode,
                content_category, content_tag, content_actor, content_director
  [Audit DB]    audit_logs
"""

import argparse
import uuid
import random
from datetime import datetime, timedelta, date

from sqlalchemy import text
from data.db import engine_audit, engine_content

# ─── Reproducibility ──────────────────────────────────────────
random.seed(42)

# ─── Enums ────────────────────────────────────────────────────

MATURITY_RATINGS = ["G", "PG", "PG-13", "R", "NC-17", "TV-Y", "TV-PG", "TV-14", "TV-MA"]
GENDERS = ["MALE", "FEMALE", "OTHER"]

# ─── Sample Data ──────────────────────────────────────────────

CATEGORIES = [
    "Action", "Comedy", "Drama", "Horror", "Sci-Fi",
    "Romance", "Thriller", "Animation", "Documentary", "Fantasy",
    "Adventure", "Mystery", "Crime", "Musical", "War",
]

TAGS = [
    "trending", "new-release", "award-winning", "classic", "must-watch",
    "binge-worthy", "feel-good", "dark", "mind-bending", "epic",
    "underrated", "cult-classic", "family-friendly", "intense", "heartwarming",
    "visually-stunning", "thought-provoking", "action-packed", "slow-burn", "twist-ending",
]

ACTORS_DATA = [
    ("Ngô Thanh Vân", "1979-02-26", "FEMALE", "Vietnamese actress and model", "Vietnamese"),
    ("Trấn Thành", "1987-02-05", "MALE", "Vietnamese comedian and actor", "Vietnamese"),
    ("Mai Thu Huyền", "1979-08-28", "FEMALE", "Vietnamese actress and filmmaker", "Vietnamese"),
    ("Lý Hải", "1973-12-10", "MALE", "Vietnamese singer, actor and director", "Vietnamese"),
    ("Ninh Dương Lan Ngọc", "1990-08-18", "FEMALE", "Vietnamese actress", "Vietnamese"),
    ("Hồng Đào", "1962-09-22", "FEMALE", "Vietnamese-American actress", "Vietnamese"),
    ("Kiều Minh Tuấn", "1987-01-07", "MALE", "Vietnamese actor", "Vietnamese"),
    ("Thu Trang", "1988-05-17", "FEMALE", "Vietnamese actress and comedian", "Vietnamese"),
    ("Dustin Nguyễn", "1962-09-17", "MALE", "Vietnamese-American actor and director", "Vietnamese"),
    ("Kaity Nguyễn", "1999-05-16", "FEMALE", "Vietnamese actress", "Vietnamese"),
    ("Isaac", "1988-10-01", "MALE", "Vietnamese singer and actor", "Vietnamese"),
    ("Jun Vũ", "1995-01-25", "FEMALE", "Vietnamese model and actress", "Vietnamese"),
    ("Trúc Anh", "1998-04-15", "FEMALE", "Vietnamese actress", "Vietnamese"),
    ("Hứa Vĩ Văn", "1981-11-15", "MALE", "Vietnamese actor and model", "Vietnamese"),
    ("Chris Evans", "1981-06-13", "MALE", "American actor known for Captain America", "American"),
    ("Scarlett Johansson", "1984-11-22", "FEMALE", "American actress and singer", "American"),
    ("Robert Downey Jr.", "1965-04-04", "MALE", "American actor known for Iron Man", "American"),
    ("Tom Holland", "1996-06-01", "MALE", "English actor known for Spider-Man", "British"),
    ("Zendaya", "1996-09-01", "FEMALE", "American actress and singer", "American"),
    ("Keanu Reeves", "1964-09-02", "MALE", "Canadian actor known for The Matrix and John Wick", "Canadian"),
    ("Ana de Armas", "1988-04-30", "FEMALE", "Cuban-Spanish actress", "Cuban"),
    ("Timothée Chalamet", "1995-12-27", "MALE", "American actor", "American"),
    ("Florence Pugh", "1996-01-03", "FEMALE", "English actress", "British"),
    ("Oscar Isaac", "1979-03-09", "MALE", "Guatemalan-American actor", "American"),
    ("Cillian Murphy", "1976-05-25", "MALE", "Irish actor known for Peaky Blinders", "Irish"),
    ("Margot Robbie", "1990-07-02", "FEMALE", "Australian actress and producer", "Australian"),
    ("Ryan Gosling", "1980-11-12", "MALE", "Canadian actor", "Canadian"),
    ("Emma Stone", "1988-11-06", "FEMALE", "American actress", "American"),
    ("Pedro Pascal", "1975-04-02", "MALE", "Chilean-American actor", "Chilean"),
    ("Song Kang-ho", "1967-01-17", "MALE", "South Korean actor", "South Korean"),
]

DIRECTORS_DATA = [
    ("Victor Vũ", "1975-08-27", "MALE", "Vietnamese-American filmmaker", "Vietnamese"),
    ("Nguyễn Quang Dũng", "1978-06-12", "MALE", "Vietnamese film director", "Vietnamese"),
    ("Trấn Thành", "1987-02-05", "MALE", "Vietnamese comedian and director", "Vietnamese"),
    ("Lý Hải", "1973-12-10", "MALE", "Vietnamese director and actor", "Vietnamese"),
    ("Phan Gia Nhật Linh", "1984-03-21", "MALE", "Vietnamese screenwriter and director", "Vietnamese"),
    ("Bong Joon-ho", "1969-09-14", "MALE", "South Korean filmmaker, won Best Picture for Parasite", "South Korean"),
    ("Christopher Nolan", "1970-07-30", "MALE", "British-American filmmaker known for Inception, Interstellar", "British"),
    ("Denis Villeneuve", "1967-10-03", "MALE", "Canadian filmmaker known for Dune, Arrival", "Canadian"),
    ("Greta Gerwig", "1983-08-04", "FEMALE", "American actress and filmmaker", "American"),
    ("Martin Scorsese", "1942-11-17", "MALE", "American filmmaker, one of the greatest directors", "American"),
    ("Quentin Tarantino", "1963-03-27", "MALE", "American filmmaker known for Pulp Fiction", "American"),
    ("Wes Anderson", "1969-05-01", "MALE", "American filmmaker known for distinct visual style", "American"),
    ("Jordan Peele", "1979-02-21", "MALE", "American actor and filmmaker known for Get Out", "American"),
    ("Chloé Zhao", "1982-03-31", "FEMALE", "Chinese filmmaker who won Best Director for Nomadland", "Chinese"),
    ("James Cameron", "1954-08-16", "MALE", "Canadian filmmaker known for Titanic, Avatar", "Canadian"),
]

# title, description, type, duration_or_none, maturity, imdb
MOVIES_DATA = [
    ("Hai Phượng", "A debt collector's daughter is kidnapped, leading her on a desperate mission across Vietnam to rescue her before it's too late.", "MOVIE", 98, "PG-13", 6.4),
    ("Bố Già", "A middle-aged father tries to balance his traditional ways with his modern family's needs in this heartfelt comedy-drama.", "MOVIE", 128, "PG", 7.2),
    ("Mắt Biếc", "A man falls deeply in love with his childhood friend, but she longs for the glittering city life, setting off a tale of unrequited love.", "MOVIE", 117, "PG", 7.5),
    ("Tiệc Trăng Máu", "A group of friends play a game of revealing their phone messages during a dinner party, exposing secrets that threaten their relationships.", "MOVIE", 120, "R", 7.0),
    ("Lật Mặt 6: Tấm Vé Định Mệnh", "An action-packed journey of family bonds and survival against all odds in rural Vietnam.", "MOVIE", 132, "PG-13", 6.8),
    ("Oppenheimer", "The story of J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II.", "MOVIE", 180, "R", 8.3),
    ("Barbie", "Barbie and Ken leave the perfection of Barbieland for the real world, where they discover the joys and perils of living among humans.", "MOVIE", 114, "PG-13", 6.9),
    ("Dune: Part Two", "Paul Atreides unites with the Fremen while on a warpath of revenge against the conspirators who destroyed his family.", "MOVIE", 166, "PG-13", 8.5),
    ("The Batman", "Batman ventures into Gotham City's underworld when a sadistic killer leaves behind a trail of cryptic clues.", "MOVIE", 176, "PG-13", 7.8),
    ("Spider-Man: No Way Home", "Spider-Man seeks help from Doctor Strange when his identity is revealed, inadvertently unleashing the multiverse.", "MOVIE", 148, "PG-13", 8.2),
    ("Top Gun: Maverick", "After more than 30 years, Maverick is still pushing the envelope as a top naval aviator.", "MOVIE", 130, "PG-13", 8.2),
    ("Everything Everywhere All at Once", "A middle-aged Chinese immigrant is swept up into an insane adventure where she alone can save existence.", "MOVIE", 139, "R", 7.8),
    ("John Wick: Chapter 4", "John Wick uncovers a path to defeating the High Table but must face a new enemy with powerful alliances.", "MOVIE", 169, "R", 7.7),
    ("Killers of the Flower Moon", "Members of the Osage Nation are murdered under mysterious circumstances in 1920s Oklahoma.", "MOVIE", 206, "R", 7.6),
    ("Poor Things", "A young woman brought back to life by an eccentric scientist runs off with a lawyer on a whirlwind adventure.", "MOVIE", 141, "R", 7.9),
    ("The Holdovers", "A cranky history teacher at a New England boarding school remains on campus during Christmas break.", "MOVIE", 133, "R", 7.9),
    ("Past Lives", "Two childhood friends reconnect after decades apart, exploring questions of fate, destiny, and love.", "MOVIE", 105, "PG-13", 7.8),
    ("Anatomy of a Fall", "A woman is suspected of her husband's death, and their blind son faces a moral dilemma as the trial unfolds.", "MOVIE", 152, "R", 7.7),
    ("Godzilla x Kong: The New Empire", "Two ancient titans, Godzilla and Kong, team up against a colossal undiscovered threat hidden within our world.", "MOVIE", 115, "PG-13", 6.4),
    ("Inside Out 2", "Riley hits puberty and encounters new emotions — Anxiety, Envy, Ennui, and Embarrassment — alongside the original gang.", "MOVIE", 100, "PG", 7.6),
    ("The Wild Robot", "A robot named Roz is stranded on an uninhabited island and must learn to adapt to the harsh wilderness.", "MOVIE", 102, "PG", 8.1),
    ("Gladiator II", "A former slave rises through the ranks of the gladiatorial arena to challenge the might of the Roman Empire.", "MOVIE", 148, "R", 6.9),
    ("Wicked", "The untold story of the witches of Oz — how a green-skinned woman became the Wicked Witch of the West.", "MOVIE", 160, "PG", 7.5),
    ("Challengers", "A former tennis prodigy turned coach transforms her husband's game by entering him in a challenger event against her ex.", "MOVIE", 131, "R", 7.6),
    ("Civil War", "A team of military-embedded journalists race across a dystopian future America during a rapidly escalating civil war.", "MOVIE", 109, "R", 7.0),
]

# title, description, type, seasons_data: [(season_num, [(ep_title, duration), ...])]
SERIES_DATA = [
    ("Thanh Sói", "A young girl grows up in the criminal underworld and becomes a legendary fighter seeking justice.", "TVSERIES", "TV-MA", 7.1,
     [(1, [("Khởi Đầu", 45), ("Bóng Tối", 48), ("Sói Con", 50), ("Đường Phố", 47), ("Trả Thù", 52), ("Phận Đời", 46)])]),
    ("Squid Game Season 2", "Gi-hun returns to the deadly game with a new purpose: to stop it once and for all.", "TVSERIES", "TV-MA", 8.0,
     [(1, [("Bread and Lottery", 55), ("Halloween Party", 50), ("001", 58), ("Six Legs", 52), ("One More Game", 54), ("O X", 48), ("Friend or Foe", 60)])]),
    ("Stranger Things 5", "The final chapter of the Hawkins crew as they face the ultimate threat from the Upside Down.", "TVSERIES", "TV-14", 8.7,
     [(1, [("The Crawl", 75), ("The Vanishing", 68), ("The Turnbow Trap", 72), ("Sorcerer", 65), ("Shock Jock", 70), ("Escape From Camazotz", 78), ("The Bridge", 80), ("The Rightside Up", 95)])]),
    ("The Last of Us Season 2", "Joel and Ellie navigate the brutal realities of a post-apocalyptic world as past decisions haunt them.", "TVSERIES", "TV-MA", 8.8,
     [(1, [("In the Weeks Ahead", 60), ("Descend", 55), ("Burn", 58), ("Would", 62), ("Intromission", 57), ("By and By", 65), ("Gut Punch", 70)])]),
    ("House of the Dragon S2", "The Targaryen civil war, known as the Dance of Dragons, intensifies as both sides prepare for an all-out battle.", "TVSERIES", "TV-MA", 8.4,
     [(1, [("A Son for a Son", 62), ("Rhaenyra the Cruel", 58), ("The Burning Mill", 65), ("The Red Dragon and the Gold", 60), ("Regent", 63), ("Smallfolk", 67), ("The Red Sowing", 70), ("The Queen Who Ever Was", 72)])]),
    ("Shōgun", "A shipwrecked English navigator becomes entangled in the ruthless political machinations of 17th century Japan.", "TVSERIES", "TV-MA", 8.7,
     [(1, [("Anjin", 70), ("Servants of Two Masters", 62), ("Tomorrow Is Tomorrow", 65), ("The Eightfold Fence", 58), ("Broken to the Fist", 60), ("Ladies of the Willow World", 63), ("A Stick of Time", 66), ("The Abyss of Life", 68), ("Crimson Sky", 72), ("A Dream of a Dream", 75)])]),
    ("Fallout", "In a post-nuclear Los Angeles, various factions fight for control of scarce resources and uncover dark secrets.", "TVSERIES", "TV-MA", 7.5,
     [(1, [("The End", 60), ("The Target", 55), ("The Head", 58), ("The Ghouls", 52), ("The Past", 57), ("The Trap", 62), ("The Radio", 59), ("The Beginning", 65)])]),
    ("3 Body Problem", "Brilliant scientists across the globe face an unprecedented alien threat that has been shaping human history from the shadows.", "TVSERIES", "TV-MA", 7.6,
     [(1, [("Countdown", 58), ("Red Coast", 55), ("Destroyer of Worlds", 60), ("Our Lord", 57), ("Judgment Day", 62), ("The Stars Our Destination", 65), ("Only Advance", 58), ("Wallfacer", 63)])]),
    ("The Bear Season 3", "Carmy continues to push for perfection as he transforms his family's sandwich shop into a fine dining experience.", "TVSERIES", "TV-MA", 8.5,
     [(1, [("Tomorrow", 35), ("Next", 32), ("Doors", 38), ("Violet", 36), ("Children", 34), ("Napkins", 33), ("Legacy", 37), ("Ice Chips", 35), ("Apologies", 40), ("Forever", 42)])]),
    ("True Detective: Night Country", "Two detectives investigate the disappearance of eight men from an Arctic research station in Ennis, Alaska.", "TVSERIES", "TV-MA", 7.2,
     [(1, [("Part 1", 60), ("Part 2", 58), ("Part 3", 55), ("Part 4", 62), ("Part 5", 57), ("Part 6", 65)])]),
    ("Slow Horses Season 4", "Failed MI5 agents at Slough House uncover a conspiracy that threatens everything they know.", "TVSERIES", "TV-MA", 8.1,
     [(1, [("Identity Theft", 48), ("A Nice Cold Pint", 45), ("Uninvited Guests", 50), ("Returns", 47), ("Grave Danger", 52), ("Hello Goodbye", 55)])]),
    ("Reacher Season 2", "Jack Reacher investigates when members of his old military unit start dying under suspicious circumstances.", "TVSERIES", "TV-MA", 7.8,
     [(1, [("ATM", 50), ("What Happens in Atlantic City", 48), ("Picture Says a Thousand Words", 52), ("A Night at the Symphony", 47), ("Burial", 55), ("New York's Finest", 50), ("The Man Goes Through", 53), ("Fly Boy", 58)])]),
    ("Silo Season 2", "Juliette ventures outside the silo to uncover the truth about the world beyond, while those inside face their own crises.", "TVSERIES", "TV-14", 8.0,
     [(1, [("The Engineer", 55), ("Order", 52), ("Solo", 50), ("Tumblers", 53), ("Descent", 48), ("Barricades", 55), ("The Dive", 58), ("The Book of Quinn", 52), ("The Safeguard", 56), ("Into the Fire", 62)])]),
    ("Wednesday Season 2", "Wednesday Addams continues her supernatural investigations and personal growth at Nevermore Academy.", "TVSERIES", "TV-14", 7.4,
     [(1, [("Here We Go Again", 48), ("Nevermore's Secrets", 45), ("The Monster Club", 50), ("Cat's in the Cradle", 47), ("Wednesday's Shadow", 52), ("Dark Side of the Moon", 55), ("Into the Darkness", 50), ("The Return", 58)])]),
    ("Loki Season 2", "Loki navigates the multiverse alongside the TVA to prevent the destruction of all timelines.", "TVSERIES", "TV-14", 8.0,
     [(1, [("Ouroboros", 52), ("Breaking Brad", 48), ("1893", 55), ("Heart of the TVA", 47), ("Science/Fiction", 58), ("Glorious Purpose", 60)])]),
    ("Arcane Season 2", "The conflict between the utopian city of Piltover and the oppressed undercity of Zaun reaches its explosive conclusion.", "TVSERIES", "TV-14", 9.0,
     [(1, [("Heavy Is the Crown", 42), ("Watch It All Burn", 40), ("Finally Got the Name Right", 45), ("Paint the Town Blue", 43), ("Blisters and Bedrock", 44), ("The Message Hidden in the Pattern", 48), ("Pretend Like It's the First Time", 46), ("Kill to the Rhythm", 50), ("The Afterlife Party", 52)])]),
    ("Severance Season 2", "Mark continues to unravel the mysteries behind the severance procedure at Lumon Industries.", "TVSERIES", "TV-MA", 8.6,
     [(1, [("Hello, Ms. Cobel", 55), ("Goodbye, Mrs. Selvig", 52), ("Who Is Alive?", 58), ("Woe's Hollow", 50), ("Trojan's Horse", 53), ("Attila", 55), ("Chikhai Bardo", 60), ("Sweet Sixteen", 52), ("The After Hours", 57), ("Cold Harbor", 65)])]),
    ("The Penguin", "Oz Cobb rises through the criminal ranks of Gotham City in the aftermath of a devastating flood.", "TVSERIES", "TV-MA", 8.2,
     [(1, [("After Hours", 55), ("Inside Man", 52), ("Bliss", 58), ("Cent'Anni", 50), ("Homecoming", 53), ("Gold Summit", 56), ("Top Hat", 60), ("A Great or Little Thing", 65)])]),
    ("Agatha All Along", "Agatha Harkness escapes a magical trap and seeks to regain her powers on the legendary Witches' Road.", "TVSERIES", "TV-14", 6.8,
     [(1, [("Seekest Thou the Road", 48), ("Circle Sewn With Fate", 45), ("Through Many Miles of Tricks and Trials", 50), ("If I Can't Reach You Let My Song Teach You", 47), ("Darkest Hour/Wake Thy Power", 55), ("Familiar By Thy Side", 52), ("Death's Hand in Mine", 50), ("Follow Me My Friend to Glory at the End", 48), ("Maiden Mother Crone", 58)])]),
    ("Black Mirror Season 7", "New standalone stories exploring the dark side of technology and its impact on human behavior.", "TVSERIES", "TV-MA", 7.5,
     [(1, [("Common People", 55), ("Bête Noire", 60), ("Plaything", 50), ("Eulogy", 58), ("Séance", 62), ("USS Callister: Into Infinity", 70)])]),
    ("Ripley", "Tom Ripley ingratiates himself into the life of a wealthy family in 1960s Italy with deadly consequences.", "TVSERIES", "TV-MA", 7.9,
     [(1, [("I Believe the Concept Is Slumming", 55), ("You're Going to Do Great", 52), ("Sommerso", 58), ("La Dolce Vita", 50), ("Lucio", 53), ("Ripley", 55), ("Macabre Entertainment", 60), ("Narcissus", 57)])]),
    ("Baby Reindeer", "A struggling comedian becomes the target of a stalker he inadvertently encouraged with kindness.", "TVSERIES", "TV-MA", 8.1,
     [(1, [("Episode 1", 30), ("Episode 2", 28), ("Episode 3", 32), ("Episode 4", 35), ("Episode 5", 30), ("Episode 6", 33), ("Episode 7", 45)])]),
    ("Disclaimer", "A journalist discovers she is the subject of a novel that exposes her darkest secrets and threatens to destroy her life.", "TVSERIES", "TV-MA", 6.6,
     [(1, [("Chapter I", 55), ("Chapter II", 52), ("Chapter III", 50), ("Chapter IV", 48), ("Chapter V", 55), ("Chapter VI", 53), ("Chapter VII", 60)])]),
    ("Pachinko Season 2", "A multigenerational Korean saga continues across Korea, Japan, and America exploring love, sacrifice, and survival.", "TVSERIES", "TV-14", 8.4,
     [(1, [("Chapter Nine", 55), ("Chapter Ten", 52), ("Chapter Eleven", 58), ("Chapter Twelve", 50), ("Chapter Thirteen", 55), ("Chapter Fourteen", 53), ("Chapter Fifteen", 57), ("Chapter Sixteen", 62)])]),
    ("Penguin Noir", "A dark reimagining of the penguin's criminal origins in the streets of Gotham's underbelly.", "TVSERIES", "TV-MA", 7.3,
     [(1, [("Shadows", 48), ("Cold Blood", 45), ("The Setup", 50), ("Double Cross", 52), ("Empire Falls", 55), ("End Game", 60)])]),
]


# Actions matching LOG_ACTION enum for user-content interactions
POSITIVE_ACTIONS_MOVIE = [
    ("PLAY_MOVIE", 2),
    ("LIKE_MOVIE", 2),
    ("ADD_MOVIE_TO_WATCHLIST", 1),
]
POSITIVE_ACTIONS_SERIES = [
    ("PLAY_EPISODE_OF_SERIES", 2),
    ("LIKE_SERIES", 2),
    ("ADD_SERIES_TO_WATCHLIST", 1),
]
NEGATIVE_ACTIONS_MOVIE = [
    ("UNLIKE_MOVIE", -1),
    ("REMOVE_MOVIE_FROM_WATCHLIST", -1),
]
NEGATIVE_ACTIONS_SERIES = [
    ("UNLIKE_SERIES", -1),
    ("REMOVE_SERIES_FROM_WATCHLIST", -1),
]

PLACEHOLDER_IMG = "https://placehold.co/400x600/1a1a2e/e94560?text={}"
PLACEHOLDER_BANNER = "https://placehold.co/1280x720/16213e/e94560?text={}"
PLACEHOLDER_TRAILER = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
PLACEHOLDER_PROFILE = "https://placehold.co/300x300/0f3460/e94560?text={}"


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


def _random_past_date(max_days: int = 365) -> datetime:
    return _now() - timedelta(days=random.randint(1, max_days))


def _random_dob(min_year: int = 1950, max_year: int = 2000) -> date:
    year = random.randint(min_year, max_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return date(year, month, day)


def _random_release_date() -> date:
    return (date.today() - timedelta(days=random.randint(30, 1095)))


# ─── Content DB Seeding ──────────────────────────────────────

def seed_content(clean: bool = False):
    """Seed content database with movies, series, categories, tags, actors, directors."""
    print("🎬 Seeding Content DB...")

    with engine_content.begin() as conn:
        if clean:
            print("  🗑️  Cleaning existing content tables...")
            for tbl in [
                "content_category", "content_tag", "content_actor", "content_director",
                "episode", "season",
                "movies", "tvseries", "content",
                "category", "tag", "actor", "director",
            ]:
                conn.execute(text(f'DROP TABLE IF EXISTS "{tbl}" CASCADE'))
            _create_content_tables(conn)

        # --- Check if tables already have data ---
        row = conn.execute(text("SELECT COUNT(*) FROM content")).fetchone()
        if row[0] > 0 and not clean:
            print(f"  ⚠️  Content table already has {row[0]} rows. Use --clean to reset.")
            return

        now = _now()

        # ─── Categories ───────────────────────────────
        cat_ids = {}
        for name in CATEGORIES:
            cid = _uuid()
            cat_ids[name] = cid
            conn.execute(text("""
                INSERT INTO category (id, "categoryName", "createdAt", "updatedAt")
                VALUES (:id, :name, :created, :updated)
            """), {"id": cid, "name": name, "created": now, "updated": now})
        print(f"  ✅ Inserted {len(CATEGORIES)} categories")

        # ─── Tags ─────────────────────────────────────
        tag_ids = {}
        for name in TAGS:
            tid = _uuid()
            tag_ids[name] = tid
            conn.execute(text("""
                INSERT INTO tag (id, "tagName", "createdAt", "updatedAt")
                VALUES (:id, :name, :created, :updated)
            """), {"id": tid, "name": name, "created": now, "updated": now})
        print(f"  ✅ Inserted {len(TAGS)} tags")

        # ─── Actors ───────────────────────────────────
        actor_ids = {}
        for name, dob_str, gender, bio, nationality in ACTORS_DATA:
            aid = _uuid()
            actor_ids[name] = aid
            dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
            conn.execute(text("""
                INSERT INTO actor (id, name, "dateOfBirth", gender, bio, "profilePicture", nationality, "createdAt", "updatedAt")
                VALUES (:id, :name, :dob, :gender, :bio, :pic, :nationality, :created, :updated)
            """), {
                "id": aid, "name": name, "dob": dob, "gender": gender,
                "bio": bio, "pic": PLACEHOLDER_PROFILE.format(name.replace(" ", "+")),
                "nationality": nationality, "created": now, "updated": now,
            })
        print(f"  ✅ Inserted {len(ACTORS_DATA)} actors")

        # ─── Directors ────────────────────────────────
        dir_ids = {}
        for name, dob_str, gender, bio, nationality in DIRECTORS_DATA:
            did = _uuid()
            dir_ids[name] = did
            dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
            conn.execute(text("""
                INSERT INTO director (id, name, "dateOfBirth", gender, bio, "profilePicture", nationality, "createdAt", "updatedAt")
                VALUES (:id, :name, :dob, :gender, :bio, :pic, :nationality, :created, :updated)
            """), {
                "id": did, "name": name, "dob": dob, "gender": gender,
                "bio": bio, "pic": PLACEHOLDER_PROFILE.format(name.replace(" ", "+")),
                "nationality": nationality, "created": now, "updated": now,
            })
        print(f"  ✅ Inserted {len(DIRECTORS_DATA)} directors")

        # ─── Movies ───────────────────────────────────
        movie_item_ids = []
        actor_names = [a[0] for a in ACTORS_DATA]
        director_names = [d[0] for d in DIRECTORS_DATA]

        for title, description, ctype, duration, maturity, imdb_rating in MOVIES_DATA:
            content_id = _uuid()
            movie_id = _uuid()
            movie_item_ids.append(movie_id)
            created = _random_past_date()
            release = _random_release_date()

            # Insert content
            conn.execute(text("""
                INSERT INTO content (id, type, title, description, "releaseDate", thumbnail, banner, trailer,
                                     "avgRating", "maturityRating", "imdbRating", "viewCount",
                                     "createdAt", "updatedAt")
                VALUES (:id, :type, :title, :desc, :release, :thumb, :banner, :trailer,
                        :avg_rating, :maturity, :imdb, :views, :created, :updated)
            """), {
                "id": content_id, "type": ctype, "title": title, "desc": description,
                "release": release,
                "thumb": PLACEHOLDER_IMG.format(title.replace(" ", "+")),
                "banner": PLACEHOLDER_BANNER.format(title.replace(" ", "+")),
                "trailer": PLACEHOLDER_TRAILER,
                "avg_rating": round(random.uniform(5.0, 9.5), 1),
                "maturity": maturity, "imdb": imdb_rating,
                "views": random.randint(1000, 500000),
                "created": created, "updated": now,
            })

            # Insert movie
            conn.execute(text("""
                INSERT INTO movies (id, duration, "content_id", "createdAt", "updatedAt")
                VALUES (:id, :duration, :cid, :created, :updated)
            """), {"id": movie_id, "duration": duration, "cid": content_id, "created": created, "updated": now})

            # Junction tables
            _insert_junctions(conn, content_id, cat_ids, tag_ids, actor_ids, dir_ids,
                              actor_names, director_names)

        print(f"  ✅ Inserted {len(MOVIES_DATA)} movies")

        # ─── TV Series + Seasons + Episodes ───────────
        series_item_ids = []
        for entry in SERIES_DATA:
            title, description, ctype, maturity, imdb_rating, seasons_data = entry

            content_id = _uuid()
            series_id = _uuid()
            series_item_ids.append(series_id)
            created = _random_past_date()
            release = _random_release_date()

            # Insert content
            conn.execute(text("""
                INSERT INTO content (id, type, title, description, "releaseDate", thumbnail, banner, trailer,
                                     "avgRating", "maturityRating", "imdbRating", "viewCount",
                                     "createdAt", "updatedAt")
                VALUES (:id, :type, :title, :desc, :release, :thumb, :banner, :trailer,
                        :avg_rating, :maturity, :imdb, :views, :created, :updated)
            """), {
                "id": content_id, "type": ctype, "title": title, "desc": description,
                "release": release,
                "thumb": PLACEHOLDER_IMG.format(title.replace(" ", "+")),
                "banner": PLACEHOLDER_BANNER.format(title.replace(" ", "+")),
                "trailer": PLACEHOLDER_TRAILER,
                "avg_rating": round(random.uniform(5.0, 9.5), 1),
                "maturity": maturity, "imdb": imdb_rating,
                "views": random.randint(5000, 1000000),
                "created": created, "updated": now,
            })

            # Insert tvseries
            conn.execute(text("""
                INSERT INTO tvseries (id, "content_id", "createdAt", "updatedAt")
                VALUES (:id, :cid, :created, :updated)
            """), {"id": series_id, "cid": content_id, "created": created, "updated": now})

            # Insert seasons & episodes
            total_eps = 0
            for season_num, episodes in seasons_data:
                season_id = _uuid()
                conn.execute(text("""
                    INSERT INTO season (id, "seasonNumber", "totalEpisodes", "tv_series_id", "createdAt", "updatedAt")
                    VALUES (:id, :snum, :total, :tsid, :created, :updated)
                """), {
                    "id": season_id, "snum": season_num, "total": len(episodes),
                    "tsid": series_id, "created": created, "updated": now,
                })

                for ep_num, (ep_title, ep_duration) in enumerate(episodes, 1):
                    ep_id = _uuid()
                    conn.execute(text("""
                        INSERT INTO episode (id, "episodeNumber", "episodeDuration", "episodeTitle",
                                             "episodeThumbnail", "season_id", "createdAt", "updatedAt")
                        VALUES (:id, :num, :dur, :title, :thumb, :sid, :created, :updated)
                    """), {
                        "id": ep_id, "num": ep_num, "dur": ep_duration, "title": ep_title,
                        "thumb": PLACEHOLDER_IMG.format(f"S{season_num}E{ep_num}"),
                        "sid": season_id, "created": created, "updated": now,
                    })
                    total_eps += 1

            # Junction tables
            _insert_junctions(conn, content_id, cat_ids, tag_ids, actor_ids, dir_ids,
                              actor_names, director_names)

        print(f"  ✅ Inserted {len(SERIES_DATA)} TV series with seasons & episodes")
        print(f"  ✅ Total content items: {len(MOVIES_DATA) + len(SERIES_DATA)}")


def _insert_junctions(conn, content_id, cat_ids, tag_ids, actor_ids, dir_ids,
                      actor_names, director_names):
    """Insert junction table rows for a content item."""
    # 2-4 random categories
    for cat in random.sample(CATEGORIES, random.randint(2, 4)):
        conn.execute(text("""
            INSERT INTO content_category (content_id, category_id) VALUES (:cid, :catid)
        """), {"cid": content_id, "catid": cat_ids[cat]})

    # 2-5 random tags
    for tag in random.sample(TAGS, random.randint(2, 5)):
        conn.execute(text("""
            INSERT INTO content_tag (content_id, tag_id) VALUES (:cid, :tagid)
        """), {"cid": content_id, "tagid": tag_ids[tag]})

    # 2-4 random actors
    for actor in random.sample(actor_names, random.randint(2, 4)):
        conn.execute(text("""
            INSERT INTO content_actor (content_id, actor_id) VALUES (:cid, :aid)
        """), {"cid": content_id, "aid": actor_ids[actor]})

    # 1-2 random directors
    for director in random.sample(director_names, random.randint(1, 2)):
        conn.execute(text("""
            INSERT INTO content_director (content_id, director_id) VALUES (:cid, :did)
        """), {"cid": content_id, "did": dir_ids[director]})


def _create_content_tables(conn):
    """Create content tables matching the exact schema."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS category (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            "categoryName" VARCHAR(255) NOT NULL,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tag (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            "tagName" VARCHAR(255) NOT NULL,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS actor (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            "dateOfBirth" DATE,
            gender VARCHAR(10),
            bio VARCHAR,
            "profilePicture" VARCHAR,
            nationality VARCHAR(250),
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS director (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            "dateOfBirth" DATE,
            gender VARCHAR(10),
            bio VARCHAR,
            "profilePicture" VARCHAR,
            nationality VARCHAR(250),
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS content (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            type VARCHAR(20) NOT NULL,
            title VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            "releaseDate" DATE NOT NULL DEFAULT CURRENT_DATE,
            thumbnail VARCHAR(250) NOT NULL,
            banner VARCHAR(250) NOT NULL,
            trailer VARCHAR(250) NOT NULL,
            "avgRating" DECIMAL(3,1) DEFAULT 0,
            "maturityRating" VARCHAR(10) DEFAULT 'G',
            "imdbRating" DECIMAL(3,1) DEFAULT 0,
            "viewCount" INTEGER DEFAULT 0,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS movies (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            duration INTEGER NOT NULL,
            "content_id" UUID UNIQUE REFERENCES content(id) ON DELETE CASCADE,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tvseries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            "content_id" UUID UNIQUE REFERENCES content(id) ON DELETE CASCADE,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS season (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            "seasonNumber" INTEGER NOT NULL,
            "totalEpisodes" INTEGER NOT NULL,
            "tv_series_id" UUID REFERENCES tvseries(id) ON DELETE CASCADE,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS episode (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            "episodeNumber" INTEGER NOT NULL,
            "episodeDuration" INTEGER NOT NULL,
            "episodeTitle" VARCHAR(250) NOT NULL,
            "episodeThumbnail" VARCHAR(250),
            "season_id" UUID REFERENCES season(id) ON DELETE CASCADE,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS content_category (
            content_id UUID REFERENCES content(id) ON DELETE CASCADE,
            category_id UUID REFERENCES category(id) ON DELETE CASCADE,
            PRIMARY KEY (content_id, category_id)
        );

        CREATE TABLE IF NOT EXISTS content_tag (
            content_id UUID REFERENCES content(id) ON DELETE CASCADE,
            tag_id UUID REFERENCES tag(id) ON DELETE CASCADE,
            PRIMARY KEY (content_id, tag_id)
        );

        CREATE TABLE IF NOT EXISTS content_actor (
            content_id UUID REFERENCES content(id) ON DELETE CASCADE,
            actor_id UUID REFERENCES actor(id) ON DELETE CASCADE,
            PRIMARY KEY (content_id, actor_id)
        );

        CREATE TABLE IF NOT EXISTS content_director (
            content_id UUID REFERENCES content(id) ON DELETE CASCADE,
            director_id UUID REFERENCES director(id) ON DELETE CASCADE,
            PRIMARY KEY (content_id, director_id)
        );
    """))
    print("  📦 Content tables created")


# ─── Audit DB Seeding ──────────────────────────────────────

def seed_audit(clean: bool = False):
    """Seed audit_logs table with realistic user interactions."""
    print("\n📊 Seeding Audit DB...")

    # First, fetch movie/series IDs from content DB
    with engine_content.connect() as conn:
        movie_ids = [
            row[0] for row in
            conn.execute(text('SELECT id FROM movies')).fetchall()
        ]
        series_ids = [
            row[0] for row in
            conn.execute(text('SELECT id FROM tvseries')).fetchall()
        ]

    if not movie_ids and not series_ids:
        print("  ❌ No content found! Seed content first.")
        return

    print(f"  📥 Found {len(movie_ids)} movies and {len(series_ids)} series in content DB")

    with engine_audit.begin() as conn:
        if clean:
            print("  🗑️  Cleaning audit_logs table...")
            conn.execute(text("DROP TABLE IF EXISTS audit_logs CASCADE"))
            _create_audit_table(conn)

        # Check existing data
        row = conn.execute(text("SELECT COUNT(*) FROM audit_logs")).fetchone()
        if row[0] > 0 and not clean:
            print(f"  ⚠️  audit_logs already has {row[0]} rows. Use --clean to reset.")
            return

        # Generate 30 fake user IDs
        user_ids = [_uuid() for _ in range(30)]
        now = _now()

        interactions = []

        for user_id in user_ids:
            session_id = _uuid()

            # Each user interacts with 8-25 items
            n_interactions = random.randint(8, 25)

            # Pick a mix of movies and series
            n_movies = random.randint(
                max(1, n_interactions // 3),
                min(len(movie_ids), n_interactions * 2 // 3)
            )
            n_series = n_interactions - n_movies

            user_movies = random.sample(movie_ids, min(n_movies, len(movie_ids)))
            user_series = random.sample(series_ids, min(n_series, len(series_ids)))

            # Generate interactions for movies
            for mid in user_movies:
                action, weight = random.choice(POSITIVE_ACTIONS_MOVIE)
                timestamp = now - timedelta(
                    days=random.randint(0, 90),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )
                interactions.append({
                    "id": _uuid(),
                    "userId": user_id,
                    "sessionId": session_id,
                    "action": action,
                    "resourceType": "MOVIE",
                    "resourceId": mid,
                    "signalWeight": weight,
                    "createdAt": timestamp,
                    "updatedAt": timestamp,
                })

            # Generate interactions for series
            for sid in user_series:
                action, weight = random.choice(POSITIVE_ACTIONS_SERIES)
                timestamp = now - timedelta(
                    days=random.randint(0, 90),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )
                interactions.append({
                    "id": _uuid(),
                    "userId": user_id,
                    "sessionId": session_id,
                    "action": action,
                    "resourceType": "SERIES",
                    "resourceId": sid,
                    "signalWeight": weight,
                    "createdAt": timestamp,
                    "updatedAt": timestamp,
                })

            # Add some negative interactions (30% chance per user)
            if random.random() < 0.3:
                n_neg = random.randint(1, 3)
                for _ in range(n_neg):
                    if random.random() < 0.5 and user_movies:
                        rid = random.choice(user_movies)
                        action, weight = random.choice(NEGATIVE_ACTIONS_MOVIE)
                        rtype = "MOVIE"
                    elif user_series:
                        rid = random.choice(user_series)
                        action, weight = random.choice(NEGATIVE_ACTIONS_SERIES)
                        rtype = "SERIES"
                    else:
                        continue

                    interactions.append({
                        "id": _uuid(),
                        "userId": user_id,
                        "sessionId": session_id,
                        "action": action,
                        "resourceType": rtype,
                        "resourceId": rid,
                        "signalWeight": weight,
                        "createdAt": now - timedelta(days=random.randint(0, 30)),
                        "updatedAt": now - timedelta(days=random.randint(0, 30)),
                    })

        # Batch insert
        for interaction in interactions:
            conn.execute(text("""
                INSERT INTO audit_logs (id, "userId", "sessionId", action, "resourceType",
                                        "resourceId", "signalWeight", "createdAt", "updatedAt")
                VALUES (:id, :userId, :sessionId, :action, :resourceType,
                        :resourceId, :signalWeight, :createdAt, :updatedAt)
            """), interaction)

        print(f"  ✅ Inserted {len(interactions)} audit log entries")
        print(f"  👤 Users: {len(user_ids)}")
        print(f"  🎬 Unique movies interacted: {len(movie_ids)}")
        print(f"  📺 Unique series interacted: {len(series_ids)}")


def _create_audit_table(conn):
    """Create audit_logs table matching the exact schema."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            "userId" VARCHAR NOT NULL,
            "sessionId" UUID NOT NULL,
            action VARCHAR(100) NOT NULL,
            "resourceType" VARCHAR(50),
            "resourceId" UUID,
            "signalWeight" SMALLINT NOT NULL DEFAULT 0,
            metadata JSONB,
            "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "deletedAt" TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs ("userId");
        CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs ("sessionId");
        CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs ("resourceId");
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs (action);
    """))
    print("  📦 audit_logs table created")


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed test data for the recommendation system")
    parser.add_argument(
        "--only",
        choices=["audit", "content"],
        help="Seed only a specific database (default: both)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop and recreate tables before seeding",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🌱 RECOMMENDATION SYSTEM — DATA SEEDER")
    print("=" * 60)

    if args.only != "audit":
        seed_content(clean=args.clean)

    if args.only != "content":
        seed_audit(clean=args.clean)

    print("\n" + "=" * 60)
    print("✅ Seeding complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
